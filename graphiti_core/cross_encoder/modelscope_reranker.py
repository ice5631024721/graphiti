"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
"""

import asyncio
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from graphiti_core.cross_encoder.client import CrossEncoderClient


class ModelScopeRerankerClient(CrossEncoderClient):
    def __init__(self, model_id: str = "BAAI/bge-reranker-v2-m3"):
        # 加载模型时应用配置：allow_remote=True（允许远程加载）
        model = Model.from_pretrained(
            model_id,
            # allow_remote=True,  # 对应配置中的 "allow_remote": true
            framework='pytorch'  # 对应配置中的 "framework": "pytorch"
        )
        # 初始化pipeline时指定任务类型（根据配置中的 "task": "text-classification"）
        self.pipeline = pipeline(
            Tasks.text_classification,  # 显式指定任务类型
            model=model  # 使用加载的模型实例
        )

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []

        input_pairs = [{'source_sentence': [query], 'sentences_to_compare': passages}]

        loop = asyncio.get_running_loop()
        formatted_input_pairs = [[query, passage] for passage in passages]

        scores_result = await loop.run_in_executor(None, self.pipeline, formatted_input_pairs)

        if isinstance(scores_result, dict) and 'scores' in scores_result:
            scores = scores_result['scores']
        elif isinstance(scores_result, list):
            scores = scores_result
        else:
            raise ValueError(f"Unexpected output format from ModelScope pipeline: {scores_result}")

        ranked_passages = sorted(
            [(passage, float(score)) for passage, score in zip(passages, scores, strict=False)],
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked_passages