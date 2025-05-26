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
    
    section_mapping = {
        "farmer": ".FARMER_DATA[]",
        "land": ".LAND_DATA[]",
        "crop": ".CROP_DATA[]",
        "weather": ".WEATHER_DATA | to_entries[] | {location: .key, data: .value[]}"
    }

    section_mapping["all"] = ".FARMER_DATA[], .LAND_DATA[], .CROP_DATA[], (.WEATHER_DATA | to_entries[] | {location: .key, data: .value[]})"

    jq_schema = tool_config.jq_schema

    if not jq_schema:
        for keyword, schema in section_mapping.items():
            if re.search(f"{keyword}s?", question.lower()):
                jq_schema = schema
                break
        
        if not jq_schema:
            jq_schema = ".[]"

    async def json_search(question: str) -> str:

        search_docs = await JSONLoader(
            file_path=tool_config.file_path,
            jq_schema=jq_schema,
            text_content=False,
            metadata_func=lambda metadata, doc: {
                "source": tool_config.file_path,
                "section": next(iter(doc)) if isinstance(doc, dict) else "unknown"
            }).aload()

        json_search_results = "\n\n---\n\n".join([
            f'<Document source="{doc.metadata["source"]}" '
            f'section="{doc.metadata.get("section", "")}">\n{json.dumps(doc.page_content, indent=2)}\n</Document>'
            for doc in search_docs
        ])

        return json_search_results


    yield FunctionInfo.from_fn(
        json_search,
        description=("""This tool retrieves relevant contexts from json search for the given question.

                        Args:
                            question (str): The question to be answered.
                    """),
    )