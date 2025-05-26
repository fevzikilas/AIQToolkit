# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig



class jsonSearchToolConfig(FunctionBaseConfig, name="json_search"):
    """
    Tool that retrieves relevant contexts from json search for the given question.
    """
    file_path: str
    jq_schema: str | None


@register_function(config_type=jsonSearchToolConfig)
async def json_search(tool_config: jsonSearchToolConfig, builder: Builder):
    from langchain_community.document_loaders import JSONLoader
    import json
    import re
    
    jq_schema = tool_config.jq_schema

    if not jq_schema:
        jq_schema = ".[]"

    async def json_search(query: str) -> str:
        search_docs = JSONLoader(
            file_path = tool_config.file_path,
            jq_schema = ".[]",
            text_content=False,
            metadata_func=lambda _, doc: {
                "source": tool_config.file_path,
                "section": next(iter(doc)) if isinstance(doc, dict) else "unknown"
            }).load()
        
        filtered_docs = [
            doc for doc in search_docs 
            if query.lower() in json.dumps(doc.page_content).lower()
        ]
        
        json_search_results = "\n".join([
            f'{json.dumps(doc.page_content, indent=3)}'
            for doc in filtered_docs
        ])
        
        
        return json_search_results




    yield FunctionInfo.from_fn(
        json_search,
        description=("""This tool retrieves relevant contexts from json search for the given question.

                        Args:
                            question (str): The question to be answered.
                    """),
    )