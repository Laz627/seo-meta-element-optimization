import streamlit as st
import pandas as pd
import openai
import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Tuple
import logging
import sys
from datetime import datetime
import traceback
import hashlib
import json
from functools import lru_cache
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from aiohttp import ClientSession, TCPConnector
from cachetools import TTLCache
import io
import openpyxl

# Configure logging to streamlit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Cache the template file creation
@st.cache_data
def create_template_file():
    """Generate an Excel template for user download."""
    template = {
        "Title Tags": pd.DataFrame({
            "URL": ["example.com/page1", "example.com/page2"],
            "Title Tag": ["Current Title 1", "Current Title 2"],
            "Primary Keyword": ["keyword 1", "keyword 2"],
            "Action": ["Reduce Length", "Add Length"]
        }),
        "H1s": pd.DataFrame({
            "URL": ["example.com/page1", "example.com/page2"],
            "H1": ["Current H1 1", "Current H1 2"],
            "Primary Keyword": ["keyword 1", "keyword 2"],
            "Action": ["Reduce Length", "Add Length"]
        }),
        "Meta Descriptions": pd.DataFrame({
            "URL": ["example.com/page1", "example.com/page2"],
            "Meta Description": ["Current Meta 1", "Current Meta 2"],
            "Primary Keyword": ["keyword 1", "keyword 2"],
            "Action": ["Reduce Length", "Add Length"]
        })
    }
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, df in template.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output

def validate_excel_structure(xlsx: Dict[str, pd.DataFrame]) -> bool:
    """Validate the structure of the uploaded Excel file."""
    required_sheets = ['Title Tags', 'H1s', 'Meta Descriptions']
    required_columns = {
        'Title Tags': ['URL', 'Title Tag', 'Action'],
        'H1s': ['URL', 'H1', 'Action'],
        'Meta Descriptions': ['URL', 'Meta Description', 'Action']
    }
    
    for sheet in required_sheets:
        if sheet not in xlsx:
            st.error(f"Missing required sheet: {sheet}")
            return False
        
        df = xlsx[sheet]
        missing_cols = [col for col in required_columns[sheet] if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns in {sheet}: {', '.join(missing_cols)}")
            return False
        
        # Validate Actions
        valid_actions = ['Reduce Length', 'Add Length', 'Create New']
        invalid_actions = df[~df['Action'].isin(valid_actions)]['Action'].unique()
        if len(invalid_actions) > 0:
            st.error(f"Invalid actions in {sheet}: {', '.join(invalid_actions)}")
            return False
    
    return True
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
            st.warning(f"Error determining page intent for URL {url}: {str(e)}")
            return 'informational'

    def get_cache_key(self, text: str, element_type: str, action: str, keyword: str, intent: str) -> str:
        data = f"{text}|{element_type}|{action}|{keyword}|{intent}"
        return hashlib.md5(data.encode()).hexdigest()

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
        try:
            intent = self.determine_page_intent(url)
            
            # Check cache
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
                
                Return only the optimized meta description, nothing else."""
            }

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
                    st.warning(f"Rate limit reached. Waiting {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                    return await self.create_optimization_request(current_text, element_type, action, keyword, url, session)
                
                result = await response.json()
                optimized_text = result['choices'][0]['message']['content'].strip()
                
                # Cache the result
                self.cache[cache_key] = optimized_text
                return optimized_text

        except Exception as e:
            st.error(f"Error optimizing {element_type} for URL {url}: {str(e)}")
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
        chunk_size = 20  # Process 20 items at a time
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
                    st.error(f"Failed to process row {index} after {max_retries} attempts")
                    return index, None

            except Exception as e:
                st.error(f"Error processing row {index}: {str(e)}")
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
                        st.error(f"Error in chunk processing: {str(result)}")

                if progress_bar:
                    progress_bar.progress((chunk_end) / len(batch_df))

            except Exception as e:
                st.error(f"Error processing chunk: {str(e)}")
                continue

            await asyncio.sleep(0.2)

        return results

        async def process_dataframe_async(
            self,
            df: pd.DataFrame,
            element_type: str,
            element_column: str,
            progress_bar=None
        ) -> pd.DataFrame:
            """Process a single DataFrame (sheet) asynchronously."""
            connector = TCPConnector(limit=20)
            timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_connect=60, sock_read=60)
    
            async with ClientSession(connector=connector, timeout=timeout) as session:
                df['New Element'] = ''
                df['New Character Length'] = 0
                df['Processing Status'] = 'Pending'
                df['Keyword Used'] = 'No Keyword Provided'
                df['Page Intent'] = df['URL'].apply(self.determine_page_intent)
    
                batch_size = 100
                total_batches = (len(df) + batch_size - 1) // batch_size
    
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
    
            return df
    
    def main():
        st.set_page_config(page_title="SEO Meta Element Optimizer", layout="wide")
        st.title("SEO Meta Element Optimizer")
        st.subheader("Optimize your website's meta elements using AI")
    
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            api_key = st.text_input("OpenAI API Key", type="password")
            brand_name = st.text_input("Brand Name")
            
            st.header("Intent Mapping")
            with st.expander("Configure URL Patterns"):
                patterns = st.text_area(
                    "Enter URL patterns (one per line) followed by intent type:",
                    "shop:transactional\n"
                    "ideas:informational\n"
                    "inspiration:inspirational\n"
                    "locations:localized\n"
                    "showroom:localized"
                )
                
                intent_mapping = {}
                for line in patterns.split('\n'):
                    if ':' in line:
                        pattern, intent = line.split(':')
                        intent_mapping[pattern.strip()] = intent.strip()
    
        # Main content
        st.markdown("""
        ### Instructions
        1. Download and fill out the template file below
        2. Enter your OpenAI API key and brand name in the sidebar
        3. Upload your completed Excel file
        4. Click 'Start Optimization' to begin processing
        """)
    
        # Template download
        template_file = create_template_file()
        st.download_button(
            label="📥 Download Template File",
            data=template_file,
            file_name="seo_meta_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
        # File upload
        uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])
    
        if uploaded_file:
            try:
                xlsx = pd.read_excel(uploaded_file, sheet_name=None)
                
                if not validate_excel_structure(xlsx):
                    st.stop()
    
                st.write("### Preview of uploaded data")
                tabs = st.tabs(['Title Tags', 'H1s', 'Meta Descriptions'])
                
                for tab, sheet_name in zip(tabs, ['Title Tags', 'H1s', 'Meta Descriptions']):
                    with tab:
                        if sheet_name in xlsx:
                            st.dataframe(xlsx[sheet_name].head())
                        else:
                            st.warning(f"Missing {sheet_name} sheet")
    
                if st.button("Start Optimization"):
                    if not api_key:
                        st.error("Please provide an OpenAI API key")
                        st.stop()
                    
                    if not brand_name:
                        st.error("Please provide a brand name")
                        st.stop()
    
                    optimizer = SEOOptimizer(
                        api_key=api_key,
                        brand_name=brand_name,
                        intent_mapping=intent_mapping
                    )
    
                    results = {}
                    progress_bars = {}
                    
                    for sheet_name in ['Title Tags', 'H1s', 'Meta Descriptions']:
                        if sheet_name not in xlsx:
                            continue
    
                        st.write(f"Processing {sheet_name}...")
                        progress_bars[sheet_name] = st.progress(0.0)
                        
                        df = xlsx[sheet_name]
                        element_type = 'title' if sheet_name == 'Title Tags' else 'h1' if sheet_name == 'H1s' else 'meta'
                        element_column = 'Title Tag' if sheet_name == 'Title Tags' else 'H1' if sheet_name == 'H1s' else 'Meta Description'
    
                        try:
                            results[sheet_name] = asyncio.run(
                                optimizer.process_dataframe_async(
                                    df,
                                    element_type,
                                    element_column,
                                    progress_bars[sheet_name]
                                )
                            )
                            st.success(f"{sheet_name} processing complete!")
                        except Exception as e:
                            st.error(f"Error processing {sheet_name}: {str(e)}")
                            continue
    
                    if results:
                        st.write("### Download Results")
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            for sheet_name, df in results.items():
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        output.seek(0)
                        st.download_button(
                            label="📥 Download Optimized Results",
                            data=output,
                            file_name=f"optimized_meta_elements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.error(traceback.format_exc())
    
    if __name__ == "__main__":
        main()
