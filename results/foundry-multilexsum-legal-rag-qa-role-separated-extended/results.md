# Benchmark Report

## System Scorecard

| system_id | quality | routing | retrieval | answer | map | query_ms | model_calls | tokens | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agentic_map | 0.7381 | 0.8229 | 0.5823 | 0.5474 | 1.0000 | 30835.3 | 568 | 3232052 | 0 |
| flat_vector | 0.5841 | 0.6750 | 0.5514 | 0.5259 | n/a | 16575.4 | 142 | 1871340 | 0 |
| hierarchical_map | 0.7174 | 0.7764 | 0.5608 | 0.5323 | 1.0000 | 30291.3 | 694 | 3773676 | 0 |
| map_reduce | 0.7293 | 0.8049 | 0.5698 | 0.5427 | 1.0000 | 34180.5 | 998 | 4914682 | 0 |
| outline_then_fill | 0.7162 | 0.7611 | 0.5736 | 0.5300 | 1.0000 | 30276.2 | 711 | 4983141 | 0 |
| refine | 0.7258 | 0.7750 | 0.5736 | 0.5546 | 1.0000 | 32661.6 | 837 | 4917873 | 0 |
| stuffing | 0.6698 | 0.6229 | 0.5917 | 0.5500 | 0.9145 | 34385.6 | 430 | 2251063 | 0 |

## Metric Means

| system_id | group | metric | mean | 95% CI | count |
| --- | --- | --- | --- | --- | --- |
| agentic_map | answer | answer_reference_token_f1 | 0.3690 | [0.3265, 0.4166] | 60 |
| agentic_map | answer | answer_reference_token_precision | 0.2731 | [0.2288, 0.3160] | 60 |
| agentic_map | answer | answer_reference_token_recall | 0.7009 | [0.6491, 0.7523] | 60 |
| agentic_map | answer | citation_precision | 0.4576 | [0.3478, 0.5697] | 60 |
| agentic_map | answer | citation_recall | 0.6333 | [0.4944, 0.7597] | 60 |
| agentic_map | answer | citation_support_rate | 0.3978 | [0.2986, 0.4932] | 60 |
| agentic_map | retrieval | context_precision_at_4 | 0.5667 | [0.4250, 0.6972] | 60 |
| agentic_map | retrieval | context_recall_at_4 | 0.5778 | [0.4389, 0.7069] | 60 |
| agentic_map | routing | document_recall_at_1 | 0.5472 | [0.4750, 0.5944] | 60 |
| agentic_map | routing | document_recall_at_3 | 0.9306 | [0.8750, 0.9778] | 60 |
| agentic_map | retrieval | evidence_quote_recall_at_4 | 0.6069 | [0.4694, 0.7417] | 60 |
| agentic_map | answer | invalid_citation_rate | 0.0000 | [0.0000, 0.0000] | 60 |
| agentic_map | map | map_completion_rate | 1.0000 | [1.0000, 1.0000] | 20 |
| agentic_map | map | map_compression_ratio | 0.0772 | [0.0584, 0.1003] | 20 |
| agentic_map | map | map_schema_validity | 1.0000 | [1.0000, 1.0000] | 20 |
| agentic_map | map | map_source_reference_validity | 1.0000 | [1.0000, 1.0000] | 20 |
| agentic_map | routing | mrr | 0.9472 | [0.8944, 0.9917] | 60 |
| agentic_map | efficiency | query_duration_ms | 30835.2711 | [28493.8690, 33438.5048] | 60 |
| agentic_map | routing | required_document_coverage | 0.8667 | [0.7667, 0.9500] | 60 |
| agentic_map | retrieval | segment_recall_at_4 | 0.5778 | [0.4514, 0.7069] | 60 |
| agentic_map | efficiency | tool_calls | 2.0000 | [2.0000, 2.0000] | 60 |
| flat_vector | answer | answer_reference_token_f1 | 0.3585 | [0.3175, 0.3986] | 60 |
| flat_vector | answer | answer_reference_token_precision | 0.2636 | [0.2272, 0.3015] | 60 |
| flat_vector | answer | answer_reference_token_recall | 0.6986 | [0.6510, 0.7382] | 60 |
| flat_vector | answer | citation_precision | 0.4028 | [0.2972, 0.5065] | 60 |
| flat_vector | answer | citation_recall | 0.6000 | [0.4569, 0.7375] | 60 |
| flat_vector | answer | citation_support_rate | 0.3577 | [0.2650, 0.4464] | 60 |
| flat_vector | retrieval | context_precision_at_4 | 0.5458 | [0.4097, 0.6764] | 60 |
| flat_vector | retrieval | context_recall_at_4 | 0.5389 | [0.4125, 0.6722] | 60 |
| flat_vector | routing | document_recall_at_1 | 0.4778 | [0.3833, 0.5556] | 60 |
| flat_vector | routing | document_recall_at_3 | 0.7889 | [0.6944, 0.8694] | 60 |
| flat_vector | retrieval | evidence_quote_recall_at_4 | 0.5819 | [0.4514, 0.7181] | 60 |
| flat_vector | answer | invalid_citation_rate | 0.0000 | [0.0000, 0.0000] | 60 |
| flat_vector | routing | mrr | 0.8500 | [0.7472, 0.9444] | 60 |
| flat_vector | efficiency | query_duration_ms | 16575.4080 | [14738.6313, 18904.1677] | 60 |
| flat_vector | routing | required_document_coverage | 0.5833 | [0.4667, 0.7000] | 60 |
| flat_vector | retrieval | segment_recall_at_4 | 0.5389 | [0.3931, 0.6639] | 60 |
| flat_vector | efficiency | tool_calls | 1.0000 | [1.0000, 1.0000] | 60 |
| hierarchical_map | answer | answer_reference_token_f1 | 0.3666 | [0.3222, 0.4147] | 60 |
| hierarchical_map | answer | answer_reference_token_precision | 0.2739 | [0.2291, 0.3245] | 60 |
| hierarchical_map | answer | answer_reference_token_recall | 0.7055 | [0.6532, 0.7558] | 60 |
| hierarchical_map | answer | citation_precision | 0.4103 | [0.2921, 0.5326] | 60 |
| hierarchical_map | answer | citation_recall | 0.6097 | [0.4528, 0.7667] | 60 |
| hierarchical_map | answer | citation_support_rate | 0.3596 | [0.2603, 0.4433] | 60 |
| hierarchical_map | retrieval | context_precision_at_4 | 0.5417 | [0.3931, 0.6778] | 60 |
| hierarchical_map | retrieval | context_recall_at_4 | 0.5611 | [0.4125, 0.6972] | 60 |
| hierarchical_map | routing | document_recall_at_1 | 0.5250 | [0.4611, 0.5833] | 60 |
| hierarchical_map | routing | document_recall_at_3 | 0.8944 | [0.8444, 0.9417] | 60 |
| hierarchical_map | retrieval | evidence_quote_recall_at_4 | 0.5792 | [0.4222, 0.7278] | 60 |
| hierarchical_map | answer | invalid_citation_rate | 0.0000 | [0.0000, 0.0000] | 60 |
| hierarchical_map | map | map_completion_rate | 1.0000 | [1.0000, 1.0000] | 20 |
| hierarchical_map | map | map_compression_ratio | 0.0740 | [0.0539, 0.0951] | 20 |
| hierarchical_map | map | map_schema_validity | 1.0000 | [1.0000, 1.0000] | 20 |
| hierarchical_map | map | map_source_reference_validity | 1.0000 | [1.0000, 1.0000] | 20 |
| hierarchical_map | routing | mrr | 0.9361 | [0.8889, 0.9806] | 60 |
| hierarchical_map | efficiency | query_duration_ms | 30291.2510 | [27134.3558, 33756.6492] | 60 |
| hierarchical_map | routing | required_document_coverage | 0.7500 | [0.6333, 0.8500] | 60 |
| hierarchical_map | retrieval | segment_recall_at_4 | 0.5611 | [0.4153, 0.7000] | 60 |
| hierarchical_map | efficiency | tool_calls | 2.0000 | [2.0000, 2.0000] | 60 |
| map_reduce | answer | answer_reference_token_f1 | 0.3734 | [0.3204, 0.4278] | 60 |
| map_reduce | answer | answer_reference_token_precision | 0.2828 | [0.2302, 0.3380] | 60 |
| map_reduce | answer | answer_reference_token_recall | 0.7110 | [0.6632, 0.7589] | 60 |
| map_reduce | answer | citation_precision | 0.4093 | [0.3059, 0.5293] | 60 |
| map_reduce | answer | citation_recall | 0.5917 | [0.4528, 0.7153] | 60 |
| map_reduce | answer | citation_support_rate | 0.4307 | [0.3282, 0.5401] | 60 |
| map_reduce | retrieval | context_precision_at_4 | 0.5514 | [0.4111, 0.6819] | 60 |
| map_reduce | retrieval | context_recall_at_4 | 0.5667 | [0.4417, 0.6889] | 60 |
| map_reduce | routing | document_recall_at_1 | 0.5417 | [0.4694, 0.5944] | 60 |
| map_reduce | routing | document_recall_at_3 | 0.9222 | [0.8778, 0.9611] | 60 |
| map_reduce | retrieval | evidence_quote_recall_at_4 | 0.5944 | [0.4653, 0.7167] | 60 |
| map_reduce | answer | invalid_citation_rate | 0.0000 | [0.0000, 0.0000] | 60 |
| map_reduce | map | map_completion_rate | 1.0000 | [1.0000, 1.0000] | 20 |
| map_reduce | map | map_compression_ratio | 0.0707 | [0.0523, 0.0905] | 20 |
| map_reduce | map | map_schema_validity | 1.0000 | [1.0000, 1.0000] | 20 |
| map_reduce | map | map_source_reference_validity | 1.0000 | [1.0000, 1.0000] | 20 |
| map_reduce | routing | mrr | 0.9389 | [0.8889, 0.9833] | 60 |
| map_reduce | efficiency | query_duration_ms | 34180.5405 | [31397.4108, 37832.8088] | 60 |
| map_reduce | routing | required_document_coverage | 0.8167 | [0.7167, 0.9000] | 60 |
| map_reduce | retrieval | segment_recall_at_4 | 0.5667 | [0.4264, 0.6875] | 60 |
| map_reduce | efficiency | tool_calls | 2.0000 | [2.0000, 2.0000] | 60 |
| outline_then_fill | answer | answer_reference_token_f1 | 0.3639 | [0.3264, 0.4025] | 60 |
| outline_then_fill | answer | answer_reference_token_precision | 0.2687 | [0.2308, 0.3105] | 60 |
| outline_then_fill | answer | answer_reference_token_recall | 0.7082 | [0.6605, 0.7526] | 60 |
| outline_then_fill | answer | citation_precision | 0.3795 | [0.2824, 0.4822] | 60 |
| outline_then_fill | answer | citation_recall | 0.5722 | [0.4347, 0.7083] | 60 |
| outline_then_fill | answer | citation_support_rate | 0.4177 | [0.3391, 0.4945] | 60 |
| outline_then_fill | retrieval | context_precision_at_4 | 0.5514 | [0.4194, 0.6792] | 60 |
| outline_then_fill | retrieval | context_recall_at_4 | 0.5722 | [0.4375, 0.7000] | 60 |
| outline_then_fill | routing | document_recall_at_1 | 0.5194 | [0.4417, 0.5806] | 60 |
| outline_then_fill | routing | document_recall_at_3 | 0.8889 | [0.8472, 0.9278] | 60 |
| outline_then_fill | retrieval | evidence_quote_recall_at_4 | 0.5986 | [0.4583, 0.7264] | 60 |
| outline_then_fill | answer | invalid_citation_rate | 0.0000 | [0.0000, 0.0000] | 60 |
| outline_then_fill | map | map_completion_rate | 1.0000 | [1.0000, 1.0000] | 20 |
| outline_then_fill | map | map_compression_ratio | 0.0720 | [0.0531, 0.0935] | 20 |
| outline_then_fill | map | map_schema_validity | 1.0000 | [1.0000, 1.0000] | 20 |
| outline_then_fill | map | map_source_reference_validity | 1.0000 | [1.0000, 1.0000] | 20 |
| outline_then_fill | routing | mrr | 0.9194 | [0.8556, 0.9722] | 60 |
| outline_then_fill | efficiency | query_duration_ms | 30276.1559 | [28021.8042, 32446.9545] | 60 |
| outline_then_fill | routing | required_document_coverage | 0.7167 | [0.6167, 0.8167] | 60 |
| outline_then_fill | retrieval | segment_recall_at_4 | 0.5722 | [0.4403, 0.6972] | 60 |
| outline_then_fill | efficiency | tool_calls | 2.0000 | [2.0000, 2.0000] | 60 |
| refine | answer | answer_reference_token_f1 | 0.3715 | [0.3274, 0.4182] | 60 |
| refine | answer | answer_reference_token_precision | 0.2778 | [0.2332, 0.3221] | 60 |
| refine | answer | answer_reference_token_recall | 0.7059 | [0.6471, 0.7545] | 60 |
| refine | answer | citation_precision | 0.4523 | [0.3526, 0.5529] | 60 |
| refine | answer | citation_recall | 0.6694 | [0.5292, 0.7903] | 60 |
| refine | answer | citation_support_rate | 0.4053 | [0.3116, 0.5126] | 60 |
| refine | retrieval | context_precision_at_4 | 0.5764 | [0.4333, 0.7028] | 60 |
| refine | retrieval | context_recall_at_4 | 0.5639 | [0.4181, 0.6958] | 60 |
| refine | routing | document_recall_at_1 | 0.5111 | [0.4472, 0.5667] | 60 |
| refine | routing | document_recall_at_3 | 0.9000 | [0.8583, 0.9417] | 60 |
| refine | retrieval | evidence_quote_recall_at_4 | 0.5903 | [0.4556, 0.7139] | 60 |
| refine | answer | invalid_citation_rate | 0.0000 | [0.0000, 0.0000] | 60 |
| refine | map | map_completion_rate | 1.0000 | [1.0000, 1.0000] | 20 |
| refine | map | map_compression_ratio | 0.0678 | [0.0510, 0.0849] | 20 |
| refine | map | map_schema_validity | 1.0000 | [1.0000, 1.0000] | 20 |
| refine | map | map_source_reference_validity | 1.0000 | [1.0000, 1.0000] | 20 |
| refine | routing | mrr | 0.9222 | [0.8750, 0.9667] | 60 |
| refine | efficiency | query_duration_ms | 32661.6179 | [28855.1820, 36770.8571] | 60 |
| refine | routing | required_document_coverage | 0.7667 | [0.6667, 0.8667] | 60 |
| refine | retrieval | segment_recall_at_4 | 0.5639 | [0.4292, 0.6903] | 60 |
| refine | efficiency | tool_calls | 2.0000 | [2.0000, 2.0000] | 60 |
| stuffing | answer | answer_reference_token_f1 | 0.3758 | [0.3397, 0.4093] | 60 |
| stuffing | answer | answer_reference_token_precision | 0.2828 | [0.2469, 0.3229] | 60 |
| stuffing | answer | answer_reference_token_recall | 0.6865 | [0.6139, 0.7494] | 60 |
| stuffing | answer | citation_precision | 0.4561 | [0.3504, 0.5582] | 60 |
| stuffing | answer | citation_recall | 0.6403 | [0.5167, 0.7639] | 60 |
| stuffing | answer | citation_support_rate | 0.4083 | [0.2959, 0.5225] | 60 |
| stuffing | retrieval | context_precision_at_4 | 0.5889 | [0.4611, 0.7139] | 60 |
| stuffing | retrieval | context_recall_at_4 | 0.5764 | [0.4472, 0.7028] | 60 |
| stuffing | routing | document_recall_at_1 | 0.4556 | [0.3611, 0.5361] | 60 |
| stuffing | routing | document_recall_at_3 | 0.7056 | [0.5778, 0.8194] | 60 |
| stuffing | retrieval | evidence_quote_recall_at_4 | 0.6250 | [0.5000, 0.7556] | 60 |
| stuffing | answer | invalid_citation_rate | 0.0000 | [0.0000, 0.0000] | 60 |
| stuffing | map | map_completion_rate | 0.7436 | [0.6207, 0.8568] | 20 |
| stuffing | map | map_compression_ratio | 0.0611 | [0.0398, 0.0857] | 20 |
| stuffing | map | map_schema_validity | 1.0000 | [1.0000, 1.0000] | 20 |
| stuffing | map | map_source_reference_validity | 1.0000 | [1.0000, 1.0000] | 20 |
| stuffing | routing | mrr | 0.8306 | [0.7167, 0.9306] | 60 |
| stuffing | efficiency | query_duration_ms | 34385.6120 | [29758.6621, 38356.1184] | 60 |
| stuffing | routing | required_document_coverage | 0.5000 | [0.3500, 0.6667] | 60 |
| stuffing | retrieval | segment_recall_at_4 | 0.5764 | [0.4375, 0.6958] | 60 |
| stuffing | efficiency | tool_calls | 2.0000 | [2.0000, 2.0000] | 60 |

## Usage Summary

| system_id | phase | kind | records | model_calls | tokens | duration_ms |
| --- | --- | --- | --- | --- | --- | --- |
| agentic_map | index | system | 20 | 388 | 2073570 | 6594780.9 |
| agentic_map | index | workflow | 20 | 0 | 0 | 6599174.7 |
| agentic_map | query | system | 60 | 180 | 1158482 | 1850116.3 |
| agentic_map | query | workflow | 62 | 0 | 0 | 1904700.0 |
| flat_vector | index | system | 20 | 22 | 988735 | 554805.4 |
| flat_vector | index | workflow | 20 | 0 | 0 | 555431.1 |
| flat_vector | query | system | 60 | 120 | 882605 | 994524.5 |
| flat_vector | query | workflow | 63 | 0 | 0 | 1027051.4 |
| hierarchical_map | index | system | 20 | 514 | 2600768 | 7869304.7 |
| hierarchical_map | index | workflow | 20 | 0 | 0 | 7876889.9 |
| hierarchical_map | query | system | 60 | 180 | 1172908 | 1817475.1 |
| hierarchical_map | query | workflow | 61 | 0 | 0 | 1835347.1 |
| map_reduce | index | system | 20 | 644 | 2651439 | 7579356.4 |
| map_reduce | index | workflow | 20 | 0 | 0 | 1399383.7 |
| map_reduce | query | system | 118 | 354 | 2263243 | 3756871.5 |
| map_reduce | query | workflow | 120 | 0 | 0 | 3790286.7 |
| outline_then_fill | index | system | 20 | 531 | 3764153 | 9037075.3 |
| outline_then_fill | index | workflow | 20 | 0 | 0 | 9089607.8 |
| outline_then_fill | query | system | 60 | 180 | 1218988 | 1816569.4 |
| outline_then_fill | query | workflow | 63 | 0 | 0 | 1882752.0 |
| refine | index | system | 20 | 528 | 2974769 | 9649576.3 |
| refine | index | workflow | 20 | 0 | 0 | 8818510.4 |
| refine | query | system | 103 | 309 | 1943104 | 3400761.4 |
| refine | query | workflow | 105 | 0 | 0 | 3437644.9 |
| stuffing | index | system | 20 | 73 | 413149 | 931073.2 |
| stuffing | index | workflow | 20 | 0 | 0 | 933014.6 |
| stuffing | query | system | 119 | 357 | 1837914 | 3475214.0 |
| stuffing | query | workflow | 120 | 0 | 0 | 3492534.5 |

## Foundry Export

- Dataset: `evaluations/foundry/dataset.jsonl`
- Manifest: `artifacts/foundry-multilexsum-legal-rag-qa-role-separated-extended/evaluations/foundry/manifest.json`
- Rows: `420`
- Systems: `agentic_map, flat_vector, hierarchical_map, map_reduce, outline_then_fill, refine, stuffing`
- Evaluators: `groundedness, relevance, retrieval, document_retrieval, response_completeness`

Managed evaluation: not available for the current export.

## Issues

- `agentic_map`: citation_precision=0.4576; citation_recall=0.6333; citation_support_rate=0.3978; document_recall_at_1=0.5472; document_recall_at_3=0.9306; mrr=0.9472; required_document_coverage=0.8667; segment_recall_at_4=0.5778
- `flat_vector`: citation_precision=0.4028; citation_recall=0.6000; citation_support_rate=0.3577; document_recall_at_1=0.4778; document_recall_at_3=0.7889; mrr=0.8500; required_document_coverage=0.5833; segment_recall_at_4=0.5389
- `hierarchical_map`: citation_precision=0.4103; citation_recall=0.6097; citation_support_rate=0.3596; document_recall_at_1=0.5250; document_recall_at_3=0.8944; mrr=0.9361; required_document_coverage=0.7500; segment_recall_at_4=0.5611
- `map_reduce`: citation_precision=0.4093; citation_recall=0.5917; citation_support_rate=0.4307; document_recall_at_1=0.5417; document_recall_at_3=0.9222; mrr=0.9389; required_document_coverage=0.8167; segment_recall_at_4=0.5667
- `outline_then_fill`: citation_precision=0.3795; citation_recall=0.5722; citation_support_rate=0.4177; document_recall_at_1=0.5194; document_recall_at_3=0.8889; mrr=0.9194; required_document_coverage=0.7167; segment_recall_at_4=0.5722
- `refine`: citation_precision=0.4523; citation_recall=0.6694; citation_support_rate=0.4053; document_recall_at_1=0.5111; document_recall_at_3=0.9000; mrr=0.9222; required_document_coverage=0.7667; segment_recall_at_4=0.5639
- `stuffing`: citation_precision=0.4561; citation_recall=0.6403; citation_support_rate=0.4083; document_recall_at_1=0.4556; document_recall_at_3=0.7056; map_completion_rate=0.7436; mrr=0.8306; required_document_coverage=0.5000; segment_recall_at_4=0.5764
