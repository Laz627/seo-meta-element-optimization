import streamlit as st
import pandas as pd
import openai
import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import sys
from datetime import datetime
import traceback
from tqdm.asyncio import tqdm_asyncio
import hashlib
import json
from functools import lru_cache
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from aiohttp import ClientSession, TCPConnector
from cachetools import TTLCache
from dotenv import load_dotenv
import os
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'seo_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class SEOOptimizer:
    def __init__(self, api_key: str, brand_name: str, examples: Dict = None, intent_mapping: Dict = None):
        self.api_key = api_key
        self.brand_name = brand_name
        self.error_logs = []
        self.processed_count = 0
        self.total_count = 0
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        self.similar_requests = defaultdict(list)
        self.session = None
        self.examples = examples or {}
        self.intent_mapping = intent_mapping or {
            'shop': 'transactional',
            'ideas': 'informational',
            'inspiration': 'inspirational',
            'locations': 'localized',
            'showroom': 'localized'
        }
        openai.api_key = self.api_key

    @lru_cache(maxsize=128)
    def determine_page_intent(self, url: str) -> str:
        try:
            for key, value in self.intent_mapping.items():
                if key in str(url).lower():
                    return value
            return 'informational'
        except Exception as e:
            logging.warning(f"Error determining page intent for URL {url}: {str(e)}")
            return 'informational'

    def get_cache_key(self, text: str, element_type: str, action: str, keyword: str, intent: str) -> str:
        data = f"{text}|{element_type}|{action}|{keyword}|{intent}"
        return hashlib.md5(data.encode()).hexdigest()

    def get_example_prompt(self, intent: str, element_type: str) -> str:
        if intent in self.examples and element_type in self.examples[intent]:
            return f"\nExample of good {element_type} for {intent} intent:\n{self.examples[intent][element_type]}"
        return ""

    @lru_cache(maxsize=128)
    def extract_existing_brand_suffix(self, title: str) -> Optional[str]:
        if not title:
            return None
        parts = title.split('|')
        if len(parts) > 1:
            return f"| {parts[-1].strip()}"
        return None

    async def create_optimization_request(
        self,
        current_text: str,
        element_type: str,
        action: str,
        keyword: Optional[str],
        url: str,
        session: ClientSession
    ) -> Optional[str]:
        intent = self.determine_page_intent(url)

        cache_key = self.get_cache_key(current_text, element_type, action, str(keyword), intent)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if pd.isna(current_text) or not isinstance(current_text, str):
            current_text = ""

        keyword_requirement = ""
        if keyword and not pd.isna(keyword) and isinstance(keyword, str):
            keyword_requirement = f"- Include primary keyword: {keyword}"

        brand_suffix = self.brand_name
        if intent == 'localized' and element_type == 'title':
            existing_suffix = self.extract_existing_brand_suffix(current_text)
            if existing_suffix:
                brand_suffix = existing_suffix.lstrip('| ')

        example_prompt = self.get_example_prompt(intent, element_type)

        prompts = {
            'title': f"""Optimize this title tag for SEO. 
            Current title: {current_text}
            Action: {action}
            Page intent: {intent}
            Brand name to append: {brand_suffix}
            
            Requirements:
            - Length as close to 65 characters as possible
            {keyword_requirement}
            - Match page intent
            - End with | {brand_suffix}
            - Maintain core meaning
            - Maintain important keyword qualifiers
            {'- Preserve existing location information' if intent == 'localized' else ''}
            {example_prompt}
            
            Return only the optimized title tag, nothing else.""",
            
            'h1': f"""Optimize this H1 tag for SEO.
            Current H1: {current_text}
            Action: {action}
            Page intent: {intent}
            
            Requirements:
            - Length as close to 65 characters as possible
            {keyword_requirement}
            - Match page intent
            - Maintain core meaning
            - Maintain important keyword qualifiers
            {'- Preserve existing location information' if intent == 'localized' else ''}
            {example_prompt}
            
            Return only the optimized H1 tag, nothing else.""",
            
            'meta': f"""Optimize this meta description for SEO.
            Current description: {current_text}
            Action: {action}
            Page intent: {intent}
            
            Requirements:
            - Length as close to 155 characters as possible
            {keyword_requirement}
            - Match page intent
            - Include clear call-to-action
            - Maintain core meaning
            - Maintain important keyword qualifiers
            {'- Preserve existing location information' if intent == 'localized' else ''}
            {example_prompt}
            
            Return only the optimized meta description, nothing else."""
        }

        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are an SEO expert. Provide only the optimized element without any additional text or explanation."},
                        {"role": "user", "content": prompts[element_type]}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 200
                }
            ) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    await asyncio.sleep(retry_after)
                    return await self.create_optimization_request(current_text, element_type, action, keyword, url, session)
                
                result = await response.json()
                optimized_text = result['choices'][0]['message']['content'].strip()

                self.cache[cache_key] = optimized_text
                return optimized_text

        except Exception as e:
            logging.error(f"Error optimizing {element_type} for URL {url}: {str(e)}")
            return None
    async def process_batch_async(
        self,
        batch_df: pd.DataFrame,
        element_type: str,
        element_column: str,
        session: ClientSession,
        progress_bar=None
    ) -> List[Tuple[int, Optional[str]]]:
        tasks = []
        results = []
        chunk_size = 20
        max_retries = 3

        async def process_with_retry(index, row, retry_count=0):
            try:
                if not self.validate_input_data(row, ['URL', 'Action']):
                    return index, None

                keyword = row.get('Primary Keyword') if 'Primary Keyword' in row else None

                result = await self.create_optimization_request(
                    current_text=row[element_column],
                    element_type=element_type,
                    action=row['Action'],
                    keyword=keyword,
                    url=row['URL'],
                    session=session
                )

                if result:
                    return index, result
                elif retry_count < max_retries:
                    await asyncio.sleep(1)
                    return await process_with_retry(index, row, retry_count + 1)
                else:
                    logging.error(f"Failed to process row {index} after {max_retries} attempts")
                    return index, None

            except Exception as e:
                logging.error(f"Error processing row {index}: {str(e)}")
                if retry_count < max_retries:
                    await asyncio.sleep(1)
                    return await process_with_retry(index, row, retry_count + 1)
                return index, None

        for chunk_start in range(0, len(batch_df), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(batch_df))
            chunk_df = batch_df.iloc[chunk_start:chunk_end]

            chunk_tasks = [
                process_with_retry(index, row)
                for index, row in chunk_df.iterrows()
            ]

            try:
                chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                for result in chunk_results:
                    if isinstance(result, tuple):
                        results.append(result)
                    else:
                        logging.error(f"Error in chunk processing: {str(result)}")

                if progress_bar:
                    progress_bar.progress((chunk_end) / len(batch_df))

            except Exception as e:
                logging.error(f"Error processing chunk: {str(e)}")
                continue

            await asyncio.sleep(0.2)

        return results

    def validate_input_data(self, row: pd.Series, required_columns: List[str]) -> bool:
        essential_columns = [col for col in required_columns if col != 'Primary Keyword']
        for column in essential_columns:
            if column not in row or pd.isna(row[column]):
                return False
        return True

    async def process_spreadsheet_async(self, file_path: str, progress_callback=None) -> bool:
        try:
            if not Path(file_path).exists():
                logging.error(f"Input file not found: {file_path}")
                return False

            try:
                xlsx = pd.read_excel(file_path, sheet_name=None)
            except Exception as e:
                logging.error(f"Error reading Excel file: {str(e)}")
                return False

            required_sheets = ['Title Tags', 'H1s', 'Meta Descriptions']
            missing_sheets = [sheet for sheet in required_sheets if sheet not in xlsx]

            if missing_sheets:
                logging.error(f"Missing required sheets: {missing_sheets}")
                return False

            connector = TCPConnector(limit=20)
            timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_connect=60, sock_read=60)

            async with ClientSession(connector=connector, timeout=timeout) as session:
                for sheet_name in required_sheets:
                    logging.info(f"Processing sheet: {sheet_name}")

                    df = xlsx[sheet_name]
                    element_type = 'title' if sheet_name == 'Title Tags' else 'h1' if sheet_name == 'H1s' else 'meta'
                    element_column = 'Title Tag' if sheet_name == 'Title Tags' else 'H1' if sheet_name == 'H1s' else 'Meta Description'

                    if element_column not in df.columns:
                        logging.error(f"Missing {element_column} column in {sheet_name}")
                        continue

                    df['New Element'] = ''
                    df['New Character Length'] = 0
                    df['Processing Status'] = 'Pending'
                    df['Keyword Used'] = 'No Keyword Provided'
                    df['Page Intent'] = df['URL'].apply(self.determine_page_intent)

                    batch_size = 100
                    total_batches = (len(df) + batch_size - 1) // batch_size

                    progress_bar = None
                    if progress_callback:
                        progress_bar = progress_callback(sheet_name)

                    for batch_num in range(0, len(df), batch_size):
                        batch_df = df.iloc[batch_num:batch_num + batch_size]
                        results = await self.process_batch_async(
                            batch_df,
                            element_type,
                            element_column,
                            session,
                            progress_bar
                        )

                        for index, result in results:
                            if result:
                                df.at[index, 'New Element'] = result
                                df.at[index, 'New Character Length'] = len(result)
                                df.at[index, 'Processing Status'] = 'Success'
                                if 'Primary Keyword' in df.columns and not pd.isna(df.at[index, 'Primary Keyword']):
                                    df.at[index, 'Keyword Used'] = 'Yes'
                            else:
                                df.at[index, 'Processing Status'] = 'Failed'

                        if batch_num % 300 == 0:
                            temp_output = f'temp_output_{sheet_name}_{batch_num}.xlsx'
                            df.to_excel(temp_output, index=False)

                        await asyncio.sleep(0.2)

                    output_file = f'optimized_meta_elements_{sheet_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
                    df.to_excel(output_file, index=False)
                    logging.info(f"Saved results for {sheet_name} to {output_file}")

            return True

        except Exception as e:
            logging.error(f"Critical error in process_spreadsheet: {str(e)}")
            logging.error(traceback.format_exc())
            return False
def setup_streamlit_page():
    st.set_page_config(page_title="SEO Meta Element Optimizer", layout="wide")
    st.title("SEO Meta Element Optimizer")

    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")
        brand_name = st.text_input("Brand Name")
        
        st.header("Intent Mapping")
        intent_mapping = {}
        
        with st.expander("Configure URL Patterns"):
            patterns = st.text_area(
                "Enter URL patterns (one per line) followed by intent type:",
                "shop:transactional\n"
                "ideas:informational\n"
                "inspiration:inspirational\n"
                "locations:localized\n"
                "showroom:localized"
            )
            
            for line in patterns.split('\n'):
                if ':' in line:
                    pattern, intent = line.split(':')
                    intent_mapping[pattern.strip()] = intent.strip()

        return api_key, brand_name, intent_mapping

def process_files(optimizer, files, progress_bars):
    results = {}
    
    for sheet_name, df in files.items():
        if sheet_name not in ['Title Tags', 'H1s', 'Meta Descriptions']:
            continue
            
        progress_bar = progress_bars[sheet_name]
        
        element_type = 'title' if sheet_name == 'Title Tags' else 'h1' if sheet_name == 'H1s' else 'meta'
        element_column = 'Title Tag' if sheet_name == 'Title Tags' else 'H1' if sheet_name == 'H1s' else 'Meta Description'
        
        if element_column not in df.columns:
            st.error(f"Missing {element_column} column in {sheet_name}")
            continue
            
        results[sheet_name] = asyncio.run(
            optimizer.process_spreadsheet_async(
                df,
                progress_callback=lambda x: progress_bar
            )
        )
    
    return results

def setup_streamlit_page():
    st.set_page_config(page_title="SEO Meta Element Optimizer", layout="wide")
    st.title("SEO Meta Element Optimizer")
    st.subheader("Created by Brandon Lazovic")

    st.markdown("""
    ## What Does This App Do?
    The **SEO Meta Element Optimizer** is designed to help you improve the on-page SEO elements of your website, such as:
    - **Title Tags**: Optimize for length, keyword inclusion, and alignment with page intent.
    - **H1 Tags**: Refine for keyword relevance and readability.
    - **Meta Descriptions**: Enhance for clarity, keyword inclusion, and click-through rate optimization.

    By leveraging OpenAI's API, the app intelligently analyzes your content and provides optimized suggestions tailored to your brand and page intent.

    ## Instructions
    1. **Enter your OpenAI API Key** in the sidebar.
    2. Upload an Excel file containing the following required sheets:
        - **Title Tags**: Columns required: `URL`, `Title Tag`, and optionally `Primary Keyword`.
        - **H1s**: Columns required: `URL`, `H1`, and optionally `Primary Keyword`.
        - **Meta Descriptions**: Columns required: `URL`, `Meta Description`, and optionally `Primary Keyword`.
    3. Press the **Start Optimization** button to process the file.
    4. Download the optimized results for your review and implementation.

    ### Notes
    - Your OpenAI API key is used securely during the session and is not stored.
    - The uploaded Excel file should have properly named sheets and columns as shown in the downloadable template.

    ### Use Cases
    - Improve SEO performance for pages with underperforming metadata.
    - Generate consistent and intent-driven meta elements for large-scale websites.
    - Save time by automating repetitive optimization tasks.

    ### Template File
    """)

    # Provide a downloadable template
    template = {
        'Title Tags': pd.DataFrame({'URL': [], 'Title Tag': [], 'Primary Keyword': []}),
        'H1s': pd.DataFrame({'URL': [], 'H1': [], 'Primary Keyword': []}),
        'Meta Descriptions': pd.DataFrame({'URL': [], 'Meta Description': [], 'Primary Keyword': []}),
    }
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in template.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name)

    st.download_button(
        label="Download Template",
        data=output.getvalue(),
        file_name="SEO_Meta_Optimization_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    with st.sidebar:
        st.header("Configuration")
        
        # Text input for the user to provide their OpenAI API key
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        
        brand_name = st.text_input("Brand Name")
        
        st.header("Intent Mapping")
        intent_mapping = {}
        
        with st.expander("Configure URL Patterns"):
            patterns = st.text_area(
                "Enter URL patterns (one per line) followed by intent type:",
                "shop:transactional\n"
                "ideas:informational\n"
                "inspiration:inspirational\n"
                "locations:localized\n"
                "showroom:localized"
            )
            
            for line in patterns.split('\n'):
                if ':' in line:
                    pattern, intent = line.split(':')
                    intent_mapping[pattern.strip()] = intent.strip()

        return api_key, brand_name, intent_mapping

def main():
    # Call setup to get user inputs
    api_key, brand_name, intent_mapping = setup_streamlit_page()
    
    # Check if the API key is provided
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        return
    
    # Set the OpenAI API key dynamically
    openai.api_key = api_key
    
    uploaded_file = st.file_uploader(
        "Upload Excel file with Title Tags, H1s, and Meta Descriptions sheets",
        type=['xlsx']
    )
    
    if uploaded_file:
        st.write("### Preview of uploaded data")
        # Rest of your code for processing the uploaded file...

# Run the Streamlit app
if __name__ == "__main__":
    main()
