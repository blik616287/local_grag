"""
Entity extraction using local LLM
"""

import json
import logging
import re
from typing import List, Dict, Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import VLLM_ENDPOINT, LLM_MODEL

logger = logging.getLogger(__name__)


class Entity:
    """Entity data class"""

    def __init__(self, name: str, entity_type: str, description: str = ""):
        self.name = name
        self.entity_type = entity_type
        self.description = description

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "type": self.entity_type,
            "description": self.description
        }


class EntityExtractor:
    """
    Extracts named entities from text using local LLM
    """

    ENTITY_EXTRACTION_PROMPT = """Extract named entities from the following text.

Identify entities of these types:
- Organization: Companies, institutions, agencies
- Person: People, including fictional characters
- Location: Cities, countries, regions, landmarks
- Technology: Programming languages, frameworks, tools, products
- Product: Commercial products, services
- Event: Conferences, releases, incidents
- Concept: Technical concepts, methodologies

For each entity, provide:
1. name: The entity name (normalized, e.g., "Apple Inc." not "apple")
2. type: One of the types listed above
3. description: Brief description of the entity's relevance in the text

Return ONLY a JSON array of entities. Example:
[
  {{"name": "Apple Inc.", "type": "Organization", "description": "Technology company mentioned as industry leader"}},
  {{"name": "Python", "type": "Technology", "description": "Programming language used in the project"}}
]

Text:
{text}

JSON array of entities:"""

    def __init__(self):
        """Initialize entity extractor with vLLM client"""
        self.client = OpenAI(
            base_url=VLLM_ENDPOINT,
            api_key="EMPTY"  # vLLM doesn't need API key
        )
        self.model = LLM_MODEL
        logger.info(f"Initialized EntityExtractor with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def extract(self, text: str, max_length: int = 3000) -> List[Entity]:
        """
        Extract entities from text

        Args:
            text: Text to extract entities from
            max_length: Maximum text length to process

        Returns:
            List of extracted entities
        """
        # Truncate if too long
        if len(text) > max_length:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_length}")
            text = text[:max_length]

        logger.info(f"Extracting entities from text ({len(text)} characters)")

        try:
            # Prepare prompt
            prompt = self.ENTITY_EXTRACTION_PROMPT.format(text=text)

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting named entities from text. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            # Parse response
            content = response.choices[0].message.content
            entities = self._parse_entities(content)

            logger.info(f"Extracted {len(entities)} entities")
            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            # Return empty list on failure rather than crashing
            return []

    def _parse_entities(self, content: str) -> List[Entity]:
        """
        Parse LLM response to extract entities

        Args:
            content: LLM response content

        Returns:
            List of Entity objects
        """
        try:
            # Try to find JSON array in the response
            # Look for content between first [ and last ]
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                json_str = content

            # Parse JSON
            entities_data = json.loads(json_str)

            if not isinstance(entities_data, list):
                logger.warning("LLM response is not a JSON array")
                return []

            # Convert to Entity objects
            entities = []
            for item in entities_data:
                if isinstance(item, dict) and 'name' in item and 'type' in item:
                    entity = Entity(
                        name=item['name'].strip(),
                        entity_type=item['type'].strip(),
                        description=item.get('description', '').strip()
                    )
                    entities.append(entity)

            return entities

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response content: {content}")
            return []
        except Exception as e:
            logger.error(f"Error parsing entities: {e}")
            return []

    def extract_batch(self, texts: List[str]) -> List[List[Entity]]:
        """
        Extract entities from multiple texts

        Args:
            texts: List of texts to process

        Returns:
            List of entity lists, one per input text
        """
        logger.info(f"Extracting entities from {len(texts)} texts")

        results = []
        for i, text in enumerate(texts):
            logger.debug(f"Processing text {i+1}/{len(texts)}")
            entities = self.extract(text)
            results.append(entities)

        return results
