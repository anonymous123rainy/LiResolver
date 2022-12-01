# LiResolver


## Introduction
Open source software (OSS) licenses regulate the conditions under which OSS can be legally reused, distributed, and modified. However, a common issue arises when incorporating third-party OSS accompanied with licenses, i.e., license incompatibility, which occurs when multiple licenses exist in one project and there are conflicts between them. Despite being problematic, fixing license incompatibility issues requires substantial efforts
due to the lack of license understanding and complex package dependency.
In this paper, we propose LiResolver, a fine-grained, scalable, and flexible tool to resolve license incompatibility issues {for open source software}. Specifically, it first understands the semantics of licenses through fine-grained entity extraction and relation extraction. Then, it detects and resolves license incompatibility issues by recommending official licenses in priority. When no official licenses can satisfy the constraints, it generates a custom license as an alternative solution. Comprehensive experiments demonstrate the effectiveness of LiResolver, with 4.09\% FP rate and 0.02\% FN rate for incompatibility issue localization, and 62.61\% of 230 real-world incompatible projects resolved by LiResolver. Furthermore, we also evaluate the impacts of license hierarchy and copyright holder detection on the effectiveness of incompatibility resolution. We discuss lessons learned and made all the datasets and the replication package of LiResolver publicly available to facilitate follow-up research.

![image](img/overview_00.png)



## Features

### Fine-grained license understanding
### Hierarchy incompatibility Detection
### License recommendation and generation



## Installation





## Example




