# MMLU-Pro coverage по доменам SOC — map `v1`

- Бенчмарк: `TIGER-Lab/MMLU-Pro`, split=`test`
- n на домен: **50**, master_seed=`20260809`
- Доменов покрыто: **22 из 22** — ни один не исключён
- Вопросов: val **1100**, test **1100** (не пересекаются)

Тиры:

- **11** × primary — профильные категории, confidence high/medium
- **8** × secondary — профильные категории, confidence low (прокси натянут)
- **3** × generic — прокси нет, срез всего MMLU-Pro

| soc_major_title | G | pref | MMLU-Pro категории | confidence | тир | n_val | n_test | прокси |
| :--- | ---: | :--- | :--- | :--- | :--- | ---: | ---: | :--- |
| Architecture and Engineering Occupations | 66 | ok | `engineering`, `physics`, `math` | high | primary | 50 | 50 | Прямое совпадение предмета: инженерия плюс её физико-математическая база. |
| Business and Financial Operations Occupations | 49 | ok | `business`, `economics` | high | primary | 50 | 50 | Прямое совпадение предмета. |
| Community and Social Service Occupations | 14 | G<15 | `psychology`, `philosophy` | medium | primary | 50 | 50 | Социальная работа опирается на психологию и этику; G=14 < 15. |
| Computer and Mathematical Occupations | 30 | ok | `computer science`, `math` | high | primary | 50 | 50 | Прямое совпадение предмета. |
| Educational Instruction and Library Occupations | 60 | ok | `psychology`, `philosophy` | medium | primary | 50 | 50 | Прокси через педагогическую психологию и гуманитарную подготовку; предметной «педагогики» в MMLU-Pro нет. |
| Healthcare Practitioners and Technical Occupations | 86 | ok | `health`, `biology` | high | primary | 50 | 50 | health в MMLU-Pro — клинические вопросы уровня практикующего врача. |
| Healthcare Support Occupations | 17 | ok | `health` | medium | primary | 50 | 50 | health — уровень практикующего врача, для support-ролей планка завышена; единственный релевантный прокси. |
| Legal Occupations | 8 | G<15 | `law` | high | primary | 50 | 50 | Маппинг надёжный, но G=8 < 15: capability мерить можно, а вот связывать её с preference по этому домену нельзя — страта слишком мала. |
| Life, Physical, and Social Science Occupations | 60 | ok | `biology`, `chemistry`, `physics`, `psychology` | high | primary | 50 | 50 | Домен покрывает все четыре естественно-научные категории напрямую. |
| Management Occupations | 52 | ok | `business`, `economics` | medium | primary | 50 | 50 | Тот же пул, что у Business and Financial Operations, но вопросы не пересекаются (disjoint_across_domains) — оценки независимы. |
| Protective Service Occupations | 29 | ok | `law` | medium | primary | 50 | 50 | law — юриспруденция, а не полевые процедуры; слабый, но содержательно связанный прокси. |
| Arts, Design, Entertainment, Sports, and Media Occupations | 43 | ok | `other`, `history`, `philosophy` | low | secondary | 50 | 50 | Категория other — сборная солянка; вместе с history и philosophy даёт гуманитарный фон, но не творческие навыки. |
| Construction and Extraction Occupations | 61 | ok | `engineering` | low | secondary | 50 | 50 | Работа по инструкции; engineering-вопросы про другую часть предметной области. |
| Farming, Fishing, and Forestry Occupations | 17 | ok | `biology` | low | secondary | 50 | 50 | Полевые сельхозработы ≠ академическая биология, но предмет смежный. |
| Installation, Maintenance, and Repair Occupations | 54 | ok | `engineering` | low | secondary | 50 | 50 | Прикладной ремонт; ближайшая академическая дисциплина — engineering. |
| Office and Administrative Support Occupations | 63 | ok | `business` | low | secondary | 50 | 50 | Административные процедуры лишь частично пересекаются с business. |
| Personal Care and Service Occupations | 32 | ok | `health` | low | secondary | 50 | 50 | Уход и сервис ≠ клиническая медицина, но предметно ближе всего к health. |
| Production Occupations | 108 | ok | `engineering`, `chemistry` | low | secondary | 50 | 50 | Производственные операции ≠ инженерная теория; прокси смежный, не прямой. |
| Sales and Related Occupations | 24 | ok | `business`, `economics` | low | secondary | 50 | 50 | Продажи как навык слабо отражены в академических business-вопросах. |
| Building and Grounds Cleaning and Maintenance Occupations | 8 | G<15 | весь MMLU-Pro | none | generic | 50 | 50 | Академического аналога нет; срез всего MMLU-Pro. G=8 < 15. |
| Food Preparation and Serving Related Occupations | 17 | ok | весь MMLU-Pro | none | generic | 50 | 50 | Академического аналога нет; срез всего MMLU-Pro. |
| Transportation and Material Moving Occupations | 53 | ok | весь MMLU-Pro | none | generic | 50 | 50 | Академического аналога нет: вождение и логистика операций не покрываются ни одной категорией. Берём срез всего MMLU-Pro — проверяем, что интервенция не ломает модель вообще, раз доменное знание проверить нечем. |

Headline capability-score — macro-average по всем 22 доменам; разбивка по тирам показывает, где просадка доменная, а где общая. Домены тира generic измеряют общую эрудицию, а не знания профессии: их просадка дублирует overall-capability и не является доменным свидетельством. Тир secondary — натянутый прокси: просадка там означает «модель хуже отвечает на смежную академическую дисциплину», а не «потеряла профессиональные знания».

Колонка **pref** отмечает домены с G < 15 (Community and Social Service Occupations, Legal Occupations, Building and Grounds Cleaning and Maintenance Occupations): capability для них измеряется полноценно, но связывать её с preference по такой страте нельзя.
