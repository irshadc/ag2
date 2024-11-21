"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[87901],{36608:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>l,contentTitle:()=>r,default:()=>u,frontMatter:()=>i,metadata:()=>s,toc:()=>c});var a=t(85893),o=t(11151);const i={custom_edit_url:"https://github.com/ag2ai/ag2/edit/main/notebook/agentchat_RetrieveChat_mongodb.ipynb",description:"Explore the use of AutoGen's RetrieveChat for tasks like code generation from docstrings, answering complex questions with human feedback, and exploiting features like Update Context, custom prompts, and few-shot learning.",source_notebook:"/notebook/agentchat_RetrieveChat_mongodb.ipynb",tags:["MongoDB","integration","RAG"],title:"Using RetrieveChat Powered by MongoDB Atlas for Retrieve Augmented Code Generation and Question Answering"},r="Using RetrieveChat Powered by MongoDB Atlas for Retrieve Augmented Code Generation and Question Answering",s={id:"notebooks/agentchat_RetrieveChat_mongodb",title:"Using RetrieveChat Powered by MongoDB Atlas for Retrieve Augmented Code Generation and Question Answering",description:"Explore the use of AutoGen's RetrieveChat for tasks like code generation from docstrings, answering complex questions with human feedback, and exploiting features like Update Context, custom prompts, and few-shot learning.",source:"@site/docs/notebooks/agentchat_RetrieveChat_mongodb.mdx",sourceDirName:"notebooks",slug:"/notebooks/agentchat_RetrieveChat_mongodb",permalink:"/ag2/docs/notebooks/agentchat_RetrieveChat_mongodb",draft:!1,unlisted:!1,editUrl:"https://github.com/ag2ai/ag2/edit/main/notebook/agentchat_RetrieveChat_mongodb.ipynb",tags:[{label:"MongoDB",permalink:"/ag2/docs/tags/mongo-db"},{label:"integration",permalink:"/ag2/docs/tags/integration"},{label:"RAG",permalink:"/ag2/docs/tags/rag"}],version:"current",frontMatter:{custom_edit_url:"https://github.com/ag2ai/ag2/edit/main/notebook/agentchat_RetrieveChat_mongodb.ipynb",description:"Explore the use of AutoGen's RetrieveChat for tasks like code generation from docstrings, answering complex questions with human feedback, and exploiting features like Update Context, custom prompts, and few-shot learning.",source_notebook:"/notebook/agentchat_RetrieveChat_mongodb.ipynb",tags:["MongoDB","integration","RAG"],title:"Using RetrieveChat Powered by MongoDB Atlas for Retrieve Augmented Code Generation and Question Answering"},sidebar:"notebooksSidebar",previous:{title:"Using RetrieveChat for Retrieve Augmented Code Generation and Question Answering",permalink:"/ag2/docs/notebooks/agentchat_RetrieveChat"},next:{title:"Using RetrieveChat Powered by PGVector for Retrieve Augmented Code Generation and Question Answering",permalink:"/ag2/docs/notebooks/agentchat_RetrieveChat_pgvector"}},l={},c=[{value:"Table of Contents",id:"table-of-contents",level:2},{value:"Set your API Endpoint",id:"set-your-api-endpoint",level:2},{value:"Construct agents for RetrieveChat",id:"construct-agents-for-retrievechat",level:2},{value:"Example 1",id:"example-1",level:3}];function d(e){const n={a:"a",admonition:"admonition",code:"code",h1:"h1",h2:"h2",h3:"h3",img:"img",li:"li",p:"p",pre:"pre",ul:"ul",...(0,o.a)(),...e.components};return(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)(n.h1,{id:"using-retrievechat-powered-by-mongodb-atlas-for-retrieve-augmented-code-generation-and-question-answering",children:"Using RetrieveChat Powered by MongoDB Atlas for Retrieve Augmented Code Generation and Question Answering"}),"\n",(0,a.jsxs)(n.p,{children:[(0,a.jsx)(n.a,{href:"https://colab.research.google.com/github/ag2ai/ag2/blob/main/notebook/agentchat_RetrieveChat_mongodb.ipynb",children:(0,a.jsx)(n.img,{src:"https://colab.research.google.com/assets/colab-badge.svg",alt:"Open In Colab"})}),"\n",(0,a.jsx)(n.a,{href:"https://github.com/ag2ai/ag2/blob/main/notebook/agentchat_RetrieveChat_mongodb.ipynb",children:(0,a.jsx)(n.img,{src:"https://img.shields.io/badge/Open%20on%20GitHub-grey?logo=github",alt:"Open on GitHub"})})]}),"\n",(0,a.jsxs)(n.p,{children:["AutoGen offers conversable agents powered by LLM, tool or human, which\ncan be used to perform tasks collectively via automated chat. This\nframework allows tool use and human participation through multi-agent\nconversation. Please find documentation about this feature\n",(0,a.jsx)(n.a,{href:"https://ag2ai.github.io/ag2/docs/Use-Cases/agent_chat",children:"here"}),"."]}),"\n",(0,a.jsxs)(n.p,{children:["RetrieveChat is a conversational system for retrieval-augmented code\ngeneration and question answering. In this notebook, we demonstrate how\nto utilize RetrieveChat to generate code and answer questions based on\ncustomized documentations that are not present in the LLM\u2019s training\ndataset. RetrieveChat uses the ",(0,a.jsx)(n.code,{children:"AssistantAgent"})," and\n",(0,a.jsx)(n.code,{children:"RetrieveUserProxyAgent"}),", which is similar to the usage of\n",(0,a.jsx)(n.code,{children:"AssistantAgent"})," and ",(0,a.jsx)(n.code,{children:"UserProxyAgent"})," in other notebooks (e.g.,\n",(0,a.jsx)(n.a,{href:"https://github.com/ag2ai/ag2/blob/main/notebook/agentchat_auto_feedback_from_code_execution.ipynb",children:"Automated Task Solving with Code Generation, Execution &\nDebugging"}),").\nEssentially, ",(0,a.jsx)(n.code,{children:"RetrieveUserProxyAgent"})," implement a different auto-reply\nmechanism corresponding to the RetrieveChat prompts."]}),"\n",(0,a.jsx)(n.h2,{id:"table-of-contents",children:"Table of Contents"}),"\n",(0,a.jsx)(n.p,{children:"We\u2019ll demonstrate six examples of using RetrieveChat for code generation\nand question answering:"}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsx)(n.li,{children:(0,a.jsx)(n.a,{href:"#example-1",children:"Example 1: Generate code based off docstrings w/o human\nfeedback"})}),"\n"]}),"\n",(0,a.jsxs)(n.admonition,{title:"Requirements",type:"info",children:[(0,a.jsx)(n.p,{children:"Some extra dependencies are needed for this notebook, which can be installed via pip:"}),(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-bash",children:"pip install autogen[retrievechat-mongodb] flaml[automl]\n"})}),(0,a.jsxs)(n.p,{children:["For more information, please refer to the ",(0,a.jsx)(n.a,{href:"/docs/installation/",children:"installation guide"}),"."]})]}),"\n",(0,a.jsxs)(n.p,{children:["Ensure you have a MongoDB Atlas instance with Cluster Tier >= M10. Read\nmore on Cluster support\n",(0,a.jsx)(n.a,{href:"https://www.mongodb.com/docs/atlas/atlas-search/manage-indexes/#create-and-manage-fts-indexes",children:"here"})]}),"\n",(0,a.jsx)(n.h2,{id:"set-your-api-endpoint",children:"Set your API Endpoint"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'import json\nimport os\n\nimport autogen\nfrom autogen import AssistantAgent\nfrom autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent\n\n# Accepted file formats for that can be stored in\n# a vector database instance\nfrom autogen.retrieve_utils import TEXT_FORMATS\n\nconfig_list = [{"model": "gpt-3.5-turbo-0125", "api_key": os.environ["OPENAI_API_KEY"], "api_type": "openai"}]\nassert len(config_list) > 0\nprint("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])\n'})}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-text",children:"models to use:  ['gpt-3.5-turbo-0125']\n"})}),"\n",(0,a.jsx)(n.admonition,{type:"tip",children:(0,a.jsxs)(n.p,{children:["Learn more about configuring LLMs for agents ",(0,a.jsx)(n.a,{href:"/docs/topics/llm_configuration",children:"here"}),"."]})}),"\n",(0,a.jsx)(n.h2,{id:"construct-agents-for-retrievechat",children:"Construct agents for RetrieveChat"}),"\n",(0,a.jsxs)(n.p,{children:["We start by initializing the ",(0,a.jsx)(n.code,{children:"AssistantAgent"})," and\n",(0,a.jsx)(n.code,{children:"RetrieveUserProxyAgent"}),". The system message needs to be set to \u201cYou are\na helpful assistant.\u201d for AssistantAgent. The detailed instructions are\ngiven in the user message. Later we will use the\n",(0,a.jsx)(n.code,{children:"RetrieveUserProxyAgent.message_generator"})," to combine the instructions\nand a retrieval augmented generation task for an initial prompt to be\nsent to the LLM assistant."]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'print("Accepted file formats for `docs_path`:")\nprint(TEXT_FORMATS)\n'})}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-text",children:"Accepted file formats for `docs_path`:\n['txt', 'json', 'csv', 'tsv', 'md', 'html', 'htm', 'rtf', 'rst', 'jsonl', 'log', 'xml', 'yaml', 'yml', 'pdf']\n"})}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'# 1. create an AssistantAgent instance named "assistant"\nassistant = AssistantAgent(\n    name="assistant",\n    system_message="You are a helpful assistant.",\n    llm_config={\n        "timeout": 600,\n        "cache_seed": 42,\n        "config_list": config_list,\n    },\n)\n\n# 2. create the RetrieveUserProxyAgent instance named "ragproxyagent"\n# Refer to https://ag2ai.github.io/ag2/docs/reference/agentchat/contrib/retrieve_user_proxy_agent\n# and https://ag2ai.github.io/ag2/docs/reference/agentchat/contrib/vectordb/mongodb\n# for more information on the RetrieveUserProxyAgent and MongoDBAtlasVectorDB\nragproxyagent = RetrieveUserProxyAgent(\n    name="ragproxyagent",\n    human_input_mode="NEVER",\n    max_consecutive_auto_reply=3,\n    retrieve_config={\n        "task": "code",\n        "docs_path": [\n            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",\n            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",\n        ],\n        "chunk_token_size": 2000,\n        "model": config_list[0]["model"],\n        "vector_db": "mongodb",  # MongoDB Atlas database\n        "collection_name": "demo_collection",\n        "db_config": {\n            "connection_string": os.environ["MONGODB_URI"],  # MongoDB Atlas connection string\n            "database_name": "test_db",  # MongoDB Atlas database\n            "index_name": "vector_index",\n            "wait_until_index_ready": 120.0,  # Setting to wait 120 seconds or until index is constructed before querying\n            "wait_until_document_ready": 120.0,  # Setting to wait 120 seconds or until document is properly indexed after insertion/update\n        },\n        "get_or_create": True,  # set to False if you don\'t want to reuse an existing collection\n        "overwrite": False,  # set to True if you want to overwrite an existing collection, each overwrite will force a index creation and reupload of documents\n    },\n    code_execution_config=False,  # set to False if you don\'t want to execute the code\n)\n'})}),"\n",(0,a.jsx)(n.h3,{id:"example-1",children:"Example 1"}),"\n",(0,a.jsx)(n.p,{children:(0,a.jsx)(n.a,{href:"#table-of-contents",children:"Back to top"})}),"\n",(0,a.jsx)(n.p,{children:"Use RetrieveChat to help generate sample code and automatically run the\ncode and fix errors if there is any."}),"\n",(0,a.jsx)(n.p,{children:"Problem: Which API should I use if I want to use FLAML for a\nclassification task and I want to train the model in 30 seconds. Use\nspark to parallel the training. Force cancel jobs if time limit is\nreached."}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'# reset the assistant. Always reset the assistant before starting a new conversation.\nassistant.reset()\n\n# given a problem, we use the ragproxyagent to generate a prompt to be sent to the assistant as the initial message.\n# the assistant receives the message and generates a response. The response will be sent back to the ragproxyagent for processing.\n# The conversation continues until the termination condition is met, in RetrieveChat, the termination condition when no human-in-loop is no code block detected.\n# With human-in-loop, the conversation will continue until the user says "exit".\ncode_problem = "How can I use FLAML to perform a classification task and use spark to do parallel training. Train 30 seconds and force cancel jobs if time limit is reached."\nchat_result = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=code_problem)\n'})}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-text",children:"2024-07-25 13:47:30,700 - autogen.agentchat.contrib.retrieve_user_proxy_agent - INFO - Use the existing collection `demo_collection`.\n2024-07-25 13:47:31,048 - autogen.agentchat.contrib.retrieve_user_proxy_agent - INFO - Found 2 chunks.\n2024-07-25 13:47:31,051 - autogen.agentchat.contrib.vectordb.mongodb - INFO - No documents to insert.\n"})}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-text",children:'Trying to create collection.\nVectorDB returns doc_ids:  [[\'bdfbc921\', \'7968cf3c\']]\nAdding content of doc bdfbc921 to context.\nAdding content of doc 7968cf3c to context.\nragproxyagent (to assistant):\n\nYou\'re a retrieve augmented coding assistant. You answer user\'s questions based on your own knowledge and the\ncontext provided by the user.\nIf you can\'t answer the question with or without the current context, you should reply exactly `UPDATE CONTEXT`.\nFor code generation, you must obey the following rules:\nRule 1. You MUST NOT install any packages because all the packages needed are already installed.\nRule 2. You must follow the formats below to write your code:\n```language\n# your code\n```\n\nUser\'s question is: How can I use FLAML to perform a classification task and use spark to do parallel training. Train 30 seconds and force cancel jobs if time limit is reached.\n\nContext is: # Integrate - Spark\n\nFLAML has integrated Spark for distributed training. There are two main aspects of integration with Spark:\n\n- Use Spark ML estimators for AutoML.\n- Use Spark to run training in parallel spark jobs.\n\n## Spark ML Estimators\n\nFLAML integrates estimators based on Spark ML models. These models are trained in parallel using Spark, so we called them Spark estimators. To use these models, you first need to organize your data in the required format.\n\n### Data\n\nFor Spark estimators, AutoML only consumes Spark data. FLAML provides a convenient function `to_pandas_on_spark` in the `flaml.automl.spark.utils` module to convert your data into a pandas-on-spark (`pyspark.pandas`) dataframe/series, which Spark estimators require.\n\nThis utility function takes data in the form of a `pandas.Dataframe` or `pyspark.sql.Dataframe` and converts it into a pandas-on-spark dataframe. It also takes `pandas.Series` or `pyspark.sql.Dataframe` and converts it into a [pandas-on-spark](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html) series. If you pass in a `pyspark.pandas.Dataframe`, it will not make any changes.\n\nThis function also accepts optional arguments `index_col` and `default_index_type`.\n\n- `index_col` is the column name to use as the index, default is None.\n- `default_index_type` is the default index type, default is "distributed-sequence". More info about default index type could be found on Spark official [documentation](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/options.html#default-index-type)\n\nHere is an example code snippet for Spark Data:\n\n```python\nimport pandas as pd\nfrom flaml.automl.spark.utils import to_pandas_on_spark\n\n# Creating a dictionary\ndata = {\n    "Square_Feet": [800, 1200, 1800, 1500, 850],\n    "Age_Years": [20, 15, 10, 7, 25],\n    "Price": [100000, 200000, 300000, 240000, 120000],\n}\n\n# Creating a pandas DataFrame\ndataframe = pd.DataFrame(data)\nlabel = "Price"\n\n# Convert to pandas-on-spark dataframe\npsdf = to_pandas_on_spark(dataframe)\n```\n\nTo use Spark ML models you need to format your data appropriately. Specifically, use [`VectorAssembler`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html) to merge all feature columns into a single vector column.\n\nHere is an example of how to use it:\n\n```python\nfrom pyspark.ml.feature import VectorAssembler\n\ncolumns = psdf.columns\nfeature_cols = [col for col in columns if col != label]\nfeaturizer = VectorAssembler(inputCols=feature_cols, outputCol="features")\npsdf = featurizer.transform(psdf.to_spark(index_col="index"))["index", "features"]\n```\n\nLater in conducting the experiment, use your pandas-on-spark data like non-spark data and pass them using `X_train, y_train` or `dataframe, label`.\n\n### Estimators\n\n#### Model List\n\n- `lgbm_spark`: The class for fine-tuning Spark version LightGBM models, using [SynapseML](https://microsoft.github.io/SynapseML/docs/features/lightgbm/about/) API.\n\n#### Usage\n\nFirst, prepare your data in the required format as described in the previous section.\n\nBy including the models you intend to try in the `estimators_list` argument to `flaml.automl`, FLAML will start trying configurations for these models. If your input is Spark data, FLAML will also use estimators with the `_spark` postfix by default, even if you haven\'t specified them.\n\nHere is an example code snippet using SparkML models in AutoML:\n\n```python\nimport flaml\n\n# prepare your data in pandas-on-spark format as we previously mentioned\n\nautoml = flaml.AutoML()\nsettings = {\n    "time_budget": 30,\n    "metric": "r2",\n    "estimator_list": ["lgbm_spark"],  # this setting is optional\n    "task": "regression",\n}\n\nautoml.fit(\n    dataframe=psdf,\n    label=label,\n    **settings,\n)\n```\n\n[Link to notebook](https://github.com/microsoft/FLAML/blob/main/notebook/automl_bankrupt_synapseml.ipynb) | [Open in colab](https://colab.research.google.com/github/microsoft/FLAML/blob/main/notebook/automl_bankrupt_synapseml.ipynb)\n\n## Parallel Spark Jobs\n\nYou can activate Spark as the parallel backend during parallel tuning in both [AutoML](/docs/Use-Cases/Task-Oriented-AutoML#parallel-tuning) and [Hyperparameter Tuning](/docs/Use-Cases/Tune-User-Defined-Function#parallel-tuning), by setting the `use_spark` to `true`. FLAML will dispatch your job to the distributed Spark backend using [`joblib-spark`](https://github.com/joblib/joblib-spark).\n\nPlease note that you should not set `use_spark` to `true` when applying AutoML and Tuning for Spark Data. This is because only SparkML models will be used for Spark Data in AutoML and Tuning. As SparkML models run in parallel, there is no need to distribute them with `use_spark` again.\n\nAll the Spark-related arguments are stated below. These arguments are available in both Hyperparameter Tuning and AutoML:\n\n- `use_spark`: boolean, default=False | Whether to use spark to run the training in parallel spark jobs. This can be used to accelerate training on large models and large datasets, but will incur more overhead in time and thus slow down training in some cases. GPU training is not supported yet when use_spark is True. For Spark clusters, by default, we will launch one trial per executor. However, sometimes we want to launch more trials than the number of executors (e.g., local mode). In this case, we can set the environment variable `FLAML_MAX_CONCURRENT` to override the detected `num_executors`. The final number of concurrent trials will be the minimum of `n_concurrent_trials` and `num_executors`.\n- `n_concurrent_trials`: int, default=1 | The number of concurrent trials. When n_concurrent_trials > 1, FLAML performes parallel tuning.\n- `force_cancel`: boolean, default=False | Whether to forcely cancel Spark jobs if the search time exceeded the time budget. Spark jobs include parallel tuning jobs and Spark-based model training jobs.\n\nAn example code snippet for using parallel Spark jobs:\n\n```python\nimport flaml\n\nautoml_experiment = flaml.AutoML()\nautoml_settings = {\n    "time_budget": 30,\n    "metric": "r2",\n    "task": "regression",\n    "n_concurrent_trials": 2,\n    "use_spark": True,\n    "force_cancel": True,  # Activating the force_cancel option can immediately halt Spark jobs once they exceed the allocated time_budget.\n}\n\nautoml.fit(\n    dataframe=dataframe,\n    label=label,\n    **automl_settings,\n)\n```\n\n[Link to notebook](https://github.com/microsoft/FLAML/blob/main/notebook/integrate_spark.ipynb) | [Open in colab](https://colab.research.google.com/github/microsoft/FLAML/blob/main/notebook/integrate_spark.ipynb)\n# Research\n\nFor technical details, please check our research publications.\n\n- [FLAML: A Fast and Lightweight AutoML Library](https://www.microsoft.com/en-us/research/publication/flaml-a-fast-and-lightweight-automl-library/). Chi Wang, Qingyun Wu, Markus Weimer, Erkang Zhu. MLSys 2021.\n\n```bibtex\n@inproceedings{wang2021flaml,\n    title={FLAML: A Fast and Lightweight AutoML Library},\n    author={Chi Wang and Qingyun Wu and Markus Weimer and Erkang Zhu},\n    year={2021},\n    booktitle={MLSys},\n}\n```\n\n- [Frugal Optimization for Cost-related Hyperparameters](https://arxiv.org/abs/2005.01571). Qingyun Wu, Chi Wang, Silu Huang. AAAI 2021.\n\n```bibtex\n@inproceedings{wu2021cfo,\n    title={Frugal Optimization for Cost-related Hyperparameters},\n    author={Qingyun Wu and Chi Wang and Silu Huang},\n    year={2021},\n    booktitle={AAAI},\n}\n```\n\n- [Economical Hyperparameter Optimization With Blended Search Strategy](https://www.microsoft.com/en-us/research/publication/economical-hyperparameter-optimization-with-blended-search-strategy/). Chi Wang, Qingyun Wu, Silu Huang, Amin Saied. ICLR 2021.\n\n```bibtex\n@inproceedings{wang2021blendsearch,\n    title={Economical Hyperparameter Optimization With Blended Search Strategy},\n    author={Chi Wang and Qingyun Wu and Silu Huang and Amin Saied},\n    year={2021},\n    booktitle={ICLR},\n}\n```\n\n- [An Empirical Study on Hyperparameter Optimization for Fine-Tuning Pre-trained Language Models](https://aclanthology.org/2021.acl-long.178.pdf). Susan Xueqing Liu, Chi Wang. ACL 2021.\n\n```bibtex\n@inproceedings{liuwang2021hpolm,\n    title={An Empirical Study on Hyperparameter Optimization for Fine-Tuning Pre-trained Language Models},\n    author={Susan Xueqing Liu and Chi Wang},\n    year={2021},\n    booktitle={ACL},\n}\n```\n\n- [ChaCha for Online AutoML](https://www.microsoft.com/en-us/research/publication/chacha-for-online-automl/). Qingyun Wu, Chi Wang, John Langford, Paul Mineiro and Marco Rossi. ICML 2021.\n\n```bibtex\n@inproceedings{wu2021chacha,\n    title={ChaCha for Online AutoML},\n    author={Qingyun Wu and Chi Wang and John Langford and Paul Mineiro and Marco Rossi},\n    year={2021},\n    booktitle={ICML},\n}\n```\n\n- [Fair AutoML](https://arxiv.org/abs/2111.06495). Qingyun Wu, Chi Wang. ArXiv preprint arXiv:2111.06495 (2021).\n\n```bibtex\n@inproceedings{wuwang2021fairautoml,\n    title={Fair AutoML},\n    author={Qingyun Wu and Chi Wang},\n    year={2021},\n    booktitle={ArXiv preprint arXiv:2111.06495},\n}\n```\n\n- [Mining Robust Default Configurations for Resource-constrained AutoML](https://arxiv.org/abs/2202.09927). Moe Kayali, Chi Wang. ArXiv preprint arXiv:2202.09927 (2022).\n\n```bibtex\n@inproceedings{kayaliwang2022default,\n    title={Mining Robust Default Configurations for Resource-constrained AutoML},\n    author={Moe Kayali and Chi Wang},\n    year={2022},\n    booktitle={ArXiv preprint arXiv:2202.09927},\n}\n```\n\n- [Targeted Hyperparameter Optimization with Lexicographic Preferences Over Multiple Objectives](https://openreview.net/forum?id=0Ij9_q567Ma). Shaokun Zhang, Feiran Jia, Chi Wang, Qingyun Wu. ICLR 2023 (notable-top-5%).\n\n```bibtex\n@inproceedings{zhang2023targeted,\n    title={Targeted Hyperparameter Optimization with Lexicographic Preferences Over Multiple Objectives},\n    author={Shaokun Zhang and Feiran Jia and Chi Wang and Qingyun Wu},\n    booktitle={International Conference on Learning Representations},\n    year={2023},\n    url={https://openreview.net/forum?id=0Ij9_q567Ma},\n}\n```\n\n- [Cost-Effective Hyperparameter Optimization for Large Language Model Generation Inference](https://arxiv.org/abs/2303.04673). Chi Wang, Susan Xueqing Liu, Ahmed H. Awadallah. ArXiv preprint arXiv:2303.04673 (2023).\n\n```bibtex\n@inproceedings{wang2023EcoOptiGen,\n    title={Cost-Effective Hyperparameter Optimization for Large Language Model Generation Inference},\n    author={Chi Wang and Susan Xueqing Liu and Ahmed H. Awadallah},\n    year={2023},\n    booktitle={ArXiv preprint arXiv:2303.04673},\n}\n```\n\n- [An Empirical Study on Challenging Math Problem Solving with GPT-4](https://arxiv.org/abs/2306.01337). Yiran Wu, Feiran Jia, Shaokun Zhang, Hangyu Li, Erkang Zhu, Yue Wang, Yin Tat Lee, Richard Peng, Qingyun Wu, Chi Wang. ArXiv preprint arXiv:2306.01337 (2023).\n\n```bibtex\n@inproceedings{wu2023empirical,\n    title={An Empirical Study on Challenging Math Problem Solving with GPT-4},\n    author={Yiran Wu and Feiran Jia and Shaokun Zhang and Hangyu Li and Erkang Zhu and Yue Wang and Yin Tat Lee and Richard Peng and Qingyun Wu and Chi Wang},\n    year={2023},\n    booktitle={ArXiv preprint arXiv:2306.01337},\n}\n```\n\n\n\n--------------------------------------------------------------------------------\nassistant (to ragproxyagent):\n\nTo use FLAML to perform a classification task and use Spark for parallel training with a timeout of 30 seconds and force canceling jobs if the time limit is reached, you can follow the below code snippet:\n\n```python\nimport flaml\nfrom flaml.automl.spark.utils import to_pandas_on_spark\nfrom pyspark.ml.feature import VectorAssembler\n\n# Prepare your data in pandas-on-spark format\ndata = {\n    "feature1": [val1, val2, val3, val4],\n    "feature2": [val5, val6, val7, val8],\n    "target": [class1, class2, class1, class2],\n}\n\ndataframe = pd.DataFrame(data)\nlabel = "target"\npsdf = to_pandas_on_spark(dataframe)\n\n# Prepare your features using VectorAssembler\ncolumns = psdf.columns\nfeature_cols = [col for col in columns if col != label]\nfeaturizer = VectorAssembler(inputCols=feature_cols, outputCol="features")\npsdf = featurizer.transform(psdf)\n\n# Define AutoML settings and fit the model\nautoml = flaml.AutoML()\nsettings = {\n    "time_budget": 30,\n    "metric": "accuracy",\n    "task": "classification",\n    "estimator_list": ["lgbm_spark"],  # Optional\n}\n\nautoml.fit(\n    dataframe=psdf,\n    label=label,\n    **settings,\n)\n```\n\nIn the code:\n- Replace `val1, val2, ..., class1, class2` with your actual data values.\n- Ensure the features and target columns are correctly specified in the data dictionary.\n- Set the `time_budget` parameter to 30 to limit the training time.\n- The `force_cancel` parameter is set to `True` to force cancel Spark jobs if the time limit is exceeded.\n\nMake sure to adapt the code to your specific dataset and requirements.\n\n--------------------------------------------------------------------------------\nragproxyagent (to assistant):\n\n\n\n--------------------------------------------------------------------------------\nassistant (to ragproxyagent):\n\nUPDATE CONTEXT\n\n--------------------------------------------------------------------------------\nUpdating context and resetting conversation.\nVectorDB returns doc_ids:  [[\'bdfbc921\', \'7968cf3c\']]\nVectorDB returns doc_ids:  [[\'bdfbc921\', \'7968cf3c\']]\nVectorDB returns doc_ids:  [[\'bdfbc921\', \'7968cf3c\']]\nVectorDB returns doc_ids:  [[\'bdfbc921\', \'7968cf3c\']]\nNo more context, will terminate.\nragproxyagent (to assistant):\n\nTERMINATE\n\n--------------------------------------------------------------------------------\n'})})]})}function u(e={}){const{wrapper:n}={...(0,o.a)(),...e.components};return n?(0,a.jsx)(n,{...e,children:(0,a.jsx)(d,{...e})}):d(e)}},11151:(e,n,t)=>{t.d(n,{Z:()=>s,a:()=>r});var a=t(67294);const o={},i=a.createContext(o);function r(e){const n=a.useContext(i);return a.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function s(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:r(e.components),a.createElement(i.Provider,{value:n},e.children)}}}]);