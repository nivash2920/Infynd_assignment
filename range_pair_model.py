import numpy as np
import os
import re
from typing import List, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False


class RangePairModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        self.training_data = []
        self.is_trained = False
        self.text_vectors = None
        
    def _extract_numbers_from_text(self, text: str) -> Optional[Tuple[int, int]]:
        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 2:
            num1, num2 = int(numbers[0]), int(numbers[1])
            return (min(num1, num2), max(num1, num2))
        return None

    def _learn_pair_pattern(self, text: str, expected_pair: Tuple[int, int]) -> None:
        range_extracted = self._extract_numbers_from_text(text)
        if range_extracted:
            start, end = range_extracted
            self.training_data.append({
                'text': text,
                'range': (start, end),
                'pair': expected_pair
            })

    def available_ranges(self) -> List[Tuple[int, int]]:
        return [item['range'] for item in self.training_data]

    def train(self, dataset: List[dict]) -> None:
        self.training_data = []
        texts = []
        for item in dataset:
            text = item['text']
            pair = tuple(item['pair'])
            self._learn_pair_pattern(text, pair)
            texts.append(text)
        
        if self.training_data:
            self.text_vectors = self.vectorizer.fit_transform(texts)
            self.is_trained = True
            print(f"Model memorized {len(self.training_data)} examples")
    
    def predict(self, text: str) -> List[Tuple[int, int]]:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        range_extracted = self._extract_numbers_from_text(text)
        if not range_extracted:
            available = ", ".join(f"{start}-{end}" for start, end in self.available_ranges())
            raise ValueError(
                "Couldn't extract a numeric range from that prompt. "
                f"Try one of the trained examples or specify a format like 'between X and Y'. "
                f"Available ranges: {available if available else 'None'}."
            )
        
        query_start, query_end = range_extracted
        similarities = self._similarity_scores(text)

        # Primary: collect stored ranges fully contained within the query range
        contained_matches: List[Tuple[float, Tuple[int, int]]] = []
        covering_matches: List[Tuple[float, Tuple[int, int]]] = []
        for idx, item in enumerate(self.training_data):
            entry_start, entry_end = item['pair']
            score = similarities[idx] if similarities.size > idx else 0.0

            if entry_start >= query_start and entry_end <= query_end:
                contained_matches.append((score, (int(entry_start), int(entry_end))))
            elif query_start >= entry_start and query_end <= entry_end:
                covering_matches.append((score, (int(entry_start), int(entry_end))))

        def _sorted_unique(matches: List[Tuple[float, Tuple[int, int]]]) -> List[Tuple[int, int]]:
            matches.sort(key=lambda x: (-x[0], x[1][0], x[1][1]))
            seen = set()
            ordered: List[Tuple[int, int]] = []
            for score, pair in matches:
                if pair not in seen and (score > 0 or not ordered):
                    seen.add(pair)
                    ordered.append(pair)
            return ordered

        contained_result = _sorted_unique(contained_matches)
        if contained_result:
            return contained_result

        covering_result = _sorted_unique(covering_matches)
        if covering_result:
            return covering_result

        # Semantic fallback: pick the closest text example and return its pair
        similar = self.find_similar_text(text, threshold=0.2)
        if similar:
            pair = similar['pair']
            return [(int(pair[0]), int(pair[1]))]

        available = ", ".join(f"{start}-{end}" for start, end in self.available_ranges())
        raise ValueError(
            f"No trained range matches the request {query_start}-{query_end}. "
            f"Available ranges: {available if available else 'None'}."
        )
    
    def find_similar_text(self, query: str, threshold: float = 0.2) -> Optional[dict]:
        """Find similar text in training data using TF-IDF similarity."""
        if not self.is_trained or len(self.training_data) == 0:
            return None
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])

        if self.text_vectors is None:
            texts = [item['text'] for item in self.training_data]
            self.text_vectors = self.vectorizer.transform(texts)

        # Cosine similarity
        similarities = (query_vector * self.text_vectors.T).toarray()[0]
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]
        
        if max_similarity >= threshold:
            return self.training_data[max_idx]
        return None

    def _similarity_scores(self, query: str) -> np.ndarray:
        if not self.is_trained or len(self.training_data) == 0:
            return np.array([])

        query_vector = self.vectorizer.transform([query])
        if self.text_vectors is None:
            texts = [item['text'] for item in self.training_data]
            self.text_vectors = self.vectorizer.transform(texts)

        return (query_vector * self.text_vectors.T).toarray()[0]
    
    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        if joblib is None:
            raise RuntimeError(
                "joblib is required to save the model. Install with: pip install joblib"
            )

        model_data = {
            'vectorizer': self.vectorizer,
            'training_data': self.training_data,
            'is_trained': self.is_trained,
            'text_vectors': self.text_vectors
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load the model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        if joblib is None:
            raise RuntimeError(
                "joblib is required to load the model. Install with: pip install joblib"
            )
        
        model_data = joblib.load(filepath)
        self.vectorizer = model_data['vectorizer']
        self.training_data = model_data['training_data']
        self.is_trained = model_data['is_trained']
        self.text_vectors = model_data.get('text_vectors')
        if self.text_vectors is None and self.is_trained:
            texts = [item['text'] for item in self.training_data]
            self.text_vectors = self.vectorizer.transform(texts)
        print(f"Model loaded from {filepath}")


class LLMIntegration:
    """Integration with Ollama LLM for query understanding."""
    
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.client = None
        if OLLAMA_AVAILABLE:
            try:
                self.client = ollama
                print(f"Ollama initialized with model: {model_name}")
            except Exception as e:
                print(f"Warning: Could not initialize Ollama: {e}")
                self.client = None
    
    def understand_query(self, query: str) -> str:
        """Use LLM to understand the query and extract range information."""
        if not self.client:
            return query  # Fallback to original query
        
        prompt = f"""You are a helpful assistant that extracts number ranges from user queries.
Given the following query, extract the two numbers that represent a range and format your response as:
"between X and Y" or "from X to Y" where X and Y are the numbers.

Query: {query}

Response (only the range, no other text):"""
        
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            return response['response'].strip()
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return query
    
    def match_query_to_dataset(self, query: str, dataset_texts: List[str]) -> Optional[str]:
        """Use LLM to find the most similar text in the dataset."""
        if not self.client:
            return None
        
        dataset_str = "\n".join([f"{i+1}. {text}" for i, text in enumerate(dataset_texts)])
        
        prompt = f"""You are a helpful assistant that matches user queries to similar examples.
Given the following query and a list of example texts, find the most similar example based on the meaning and intent.

Query: {query}

Examples:
{dataset_str}

Respond with only the number (1-{len(dataset_texts)}) of the most similar example, or "none" if no good match:"""
        
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            result = response['response'].strip()
            # Try to extract number
            match = re.search(r'\d+', result)
            if match:
                idx = int(match.group()) - 1
                if 0 <= idx < len(dataset_texts):
                    return dataset_texts[idx]
            return None
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None
    
    def process_query_with_llm(self, query: str, model: 'RangePairModel', dataset: List[dict]) -> List[Tuple[int, int]]:
        """Process a query using LLM to understand and match with dataset."""
        if not self.client:
            # Fallback to direct prediction
            return model.predict(query)
        
        dataset_texts = [item['text'] for item in dataset]
        
        # Try to match with dataset using LLM
        matched_text = self.match_query_to_dataset(query, dataset_texts)
        
        if matched_text:
            # Find the pairs for matched text
            for item in dataset:
                if item['text'] == matched_text:
                    # Check if ranges match
                    query_range = model._extract_numbers_from_text(query)
                    dataset_range = model._extract_numbers_from_text(matched_text)
                    
                    if query_range and dataset_range and query_range == dataset_range:
                        # Exact match - return exact pair
                        pair = item['pair']
                        return [(int(pair[0]), int(pair[1]))]
                    elif query_range and dataset_range:
                        # Different range, but similar query - use model to predict
                        return model.predict(query)
                    else:
                        pair = item['pair']
                        return [(int(pair[0]), int(pair[1]))]
        
        # No match found, use LLM to understand query and then predict
        understood_query = self.understand_query(query)
        return model.predict(understood_query)


def create_sample_dataset() -> List[dict]:
    """Create the sample dataset."""
    return [
        {
            'text': 'Give me numbers between 5 and 20',
            'pair': (5, 20)
        },
        {
            'text': 'Values 200 to 1000',
            'pair': (200, 1000)
        },
        {
            'text': 'List everything from 50 through 150',
            'pair': (50, 150)
        },
        {
            'text': 'Need range between 100 and 300',
            'pair': (100, 300)
        },
        {
            'text': 'Show span from 600 to 1200',
            'pair': (600, 1200)
        },
        {
            'text': 'Provide interval 20 to 80',
            'pair': (20, 80)
        },
        {
            'text': 'Between 1500 and 2000',
            'pair': (1500, 2000)
        },
        {
            'text': 'Numbers from 350 to 450',
            'pair': (350, 450)
        }
    ]


def main():
    """Train the model and interactively answer user prompts."""
    dataset = create_sample_dataset()
    print("Sample dataset loaded with the following prompts and expected ranges:")
    for item in dataset:
        print(f"  - {item['text']} -> {item['pair']}")
    print()

    model = RangePairModel()

    print("Training model...")
    model.train(dataset)
    active_model = model

    model_path = "range_pair_model.pkl"
    if joblib is not None:
        try:
            model.save(model_path)
            print("Reloading model from disk...")
            reloaded_model = RangePairModel()
            reloaded_model.load(model_path)
            active_model = reloaded_model
        except RuntimeError as exc:
            print(f"Skipping save/load step: {exc}")
    else:
        print("joblib not available; skipping save/load demonstration.")

    llm = LLMIntegration() if OLLAMA_AVAILABLE else None
    if not OLLAMA_AVAILABLE:
        print(
            "Ollama integration not available. Install with `pip install ollama` "
            "and ensure the Ollama service is running for enhanced understanding."
        )

    print("\nEnter range-related prompts (type 'exit' to quit).")
    while True:
        user_query = input("\nYour prompt: ").strip()
        if not user_query:
            print("Please enter a prompt or type 'exit' to finish.")
            continue
        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            if llm and llm.client:
                pairs = llm.process_query_with_llm(user_query, active_model, dataset)
            else:
                pairs = active_model.predict(user_query)

            if pairs:
                print(f"Pairs covering the requested range: {pairs}")
            else:
                print("Sorry, I couldn't infer a numeric range from that prompt.")
        except ValueError as exc:
            print(exc)
        except RuntimeError as exc:
            print(f"Runtime error: {exc}")


if __name__ == "__main__":
    main()

