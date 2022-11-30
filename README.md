------------
#Sentiment Analysis for Amazon Multilingual Review Corpus using Bert Based Models
####Author: Tong Luo
####Date: 10/31/2022
####Update: 11/29/2022
------------
### Introduction to Sentiment Analysis

Sentiment analysis is an Machine Learning technology that extracts meanings from language input, e.g. text, voice. It can be classified into following categories:
* a. Satisfaction rating, binary (e.. satisfied, not satisfied), or customer ratings (e.g. satisfaction range in [-1, 0, 1] or [1 to 5])
* b. Emotion detection (e.g. happy, sad, mad, shock, grief, frustracted, etc.)
* c. Fine grained analysis (e.g, topics, subtopics)
* d. Intent based analysis (e.g. "The product is fine, but I need to change batter often")
* e. Aspect based analysis (e.g. 'This hammock's nylon fabric is lightweight and easy to clean, but a little scratch can totally ruin it').

Sentiment analysis have wide applications in broad variety of business. It products very import information to business decision makers because it can be used to find out brand popularity, customer satisfaction, need for customer service, demographic and market analysis, marketing ROI returns et al.  

Tradition machine learning methods use regression, classification, feature engineering methods. It is able to produce rating classifications but not perform well to catch semantic subtleties such as sarcasms in sentences, and misinterpret user's true intention. ["Attention"](https://drive.google.com/file/d/1ZnnayrP7Ue3gRWl7EycHRJuEelkUzNOo/view?usp=sharing) concept was proposed by Bahdanau[19](https://arxiv.org/abs/1409.0473),Minh-Tangh[20](https://arxiv.org/abs/1508.04025) to catch the context information for each word in sentences. Recent ["transformer"](https://drive.google.com/file/d/1LsT-esnOO0fdUcNviaGS68b-GRyjqywU/view?usp=sharing) [1](https://arxiv.org/pdf/1706.03762.pdf) based NLP models, such as Bert based models [2](https://arxiv.org/abs/1810.04805v2),[3](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment), using [self and multi-headed attention](https://drive.google.com/file/d/1Ttdw1sST-w3R8qLOZGhFNHszBaySOKfz/view?usp=sharing), and [encoder-decoder attention](https://drive.google.com/file/d/1Oqz-RxpDJ_RyMjtzPpj65p4PYdcW9q4O/view?usp=sharing) mechanism [1](https://arxiv.org/pdf/1706.03762.pdf) have achieved significant higher accuracy in language understanding and shows promising new landscape for NLU applications.

###T. Luo's works for this project
------------
In this assignment I did sentiment analysis on Multilingual Amazon Revieew Corpus (MARC) dataset. The work is presented in following order:
  1. Data validation and visualization: check invalid data and data distribution.
  2. Using nlptown/bert-base-multilingual-uncased-sentiment model[3](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) on MARC to produce user rating .This model was finetuned for sentiment analysis on product reviews in six languages: English, Dutch, German, French, Spanish and Italian. It predicts the sentiment of the review as a number of stars (between 1 and 5).
  3. The initial result from the Bert-based system shows good accuracy (63.72%) on whole test set for 5 categories on MARC data set. This Bert based model was not trained with Amazon dataset.
  4. Literatures [12](https://arxiv.org/abs/1911.02116),[13](https://towardsdatascience.com/), shows RoBerta based has higher (~30%) accuracy performance. I find a lewtun/XML Roberta model from hugging face [4][5], which was trained on Amazon MARC and did a comparison with Bert based model. Surprisingly, the lewtun/XML-RoBERTa model produced lower accuracy (achieved 55.96%) on whole test set than the Bert based model. I found out that simply padding to max-length can cause the model to lose focus significantly. I did some research and found "smart padding" [17](https://towardsdatascience.com/multilingual-amazon-reviews-classification-f55f8a650c9a) can improve sentimental accuracy.  
  5. I also found out that differnt feature engineering can affect accuracy. Combining review_title and product_category with review_body can improve the accuracy. The order of the feature matters too. Best accuracy is achieved by puting review_title at first place, product_category at sencond place, and review_body at third place. I used "smart padding" together this multi-feature engineering technology in my analysis and achieved 64.62% accuracy on whle test set, significantly better than the [Keung's paper](https://arxiv.org/abs/2010.02573) that introduced Amazon MARC dataset.
  6. I did hyperparameter optimization (HPO) using Optuna with default parameters.
  7. Retrain the XML RoBERTa based model with the best run_parameters and obtain v4 version, and achieves higher accuracy (65.02%).
  8. publish V4 to [Hugingface](https://huggingface.co/tlttl/tluo_xml_roberta_base_amazon_review_sentiment_v4). Please feel free and try it there. 
------------
##Tong's model win!
------------
* 1st Place: Tong's XML_RoBerta model           (0.654)
* 2nd Place: NLPTown's Bert_base_Model          (0.627)
* 3rd Place: Lewtun's XML_RoBerta_base          (0.582)
------------
  9. I learned to used huggingface trainer to do the training job this, and published the trained model on hugging face for public [trail and download](https://huggingface.co/tlttl/tluo_xml_roberta_base_amazon_review_sentiment_v3) [18](https://huggingface.co/tlttl/tluo_xml_roberta_base_amazon_review_sentiment_v3) .
  10. The original Amazon MARC dataset has 1.2M rows in training set, and 30k rows for validation and testing set. To speed up the training, I use dataset sharding technology, train the model on 1/10 of training, and 1/5 of validation data. Total training time is about 2.58 hours.
  11. To fine tune the model, I used Auto hyper parameter search of this model and obtained optimized hyperparameter set.
  12. Study Transformer and BERT architecture. Here is my [summary](https://drive.google.com/file/d/1Og0Ip334e5lDgvsSogejZ2l5FXtvVDsG/view?usp=sharing) and related [complexity analysis numbers](https://drive.google.com/file/d/1WOexPGcpbl9J3CtnRgMwGJWCyEOmCNyR/view?usp=sharing).

###Future works:
------------
Transformer and Bert based models have some drawbacks such as difficult to explain the results, have large parameter size, etc. Attention mechanism of Bert provides an opportunity to fine tune our focus (i.e attention) to the right wordss in sentences, and hence opens up new areas of application such as fine-grain, aspect, emotion and intent analysis.  
Akbar et.al [6](https://arxiv.org/abs/2001.11316) proposed BERT Adversarial Training (BAT) model for aspect based semantic analysis. Manish et.al proposed fine grained sentimental analysis [7](https://ieeexplore.ieee.org/document/8947435). Qian Chen etal. proposed joint intention based Bert[8](https://arxiv.org/pdf/1902.10909.pdf). Chiorrini et.al proposed ways to fine tune Bert model to achieve good emotion prediction[9](https://www.researchgate.net/publication/350591267_Emotion_and_sentiment_analysis_of_tweets_using_BERT). Adoma et.al compared emotion analysis of several Bert based models [10](https://ieeexplore.ieee.org/abstract/document/9317379). These technologies can be very helpful to Amazon's busines.

Many research works are done to reduce BERT parameter size and to speed up the training process. Recent works can be found in DistilBert [14](https://arxiv.org/abs/1910.01108), TinyBert [15](https://dair.ai/TinyBERT-Size_does_matter,_but_how_you_train_it_can_be_more_important/), MiniML [16](https://arxiv.org/abs/2002.10957). The new models can reduce parameter size 30~50%, yet still maintain comparable performance. 

Bert was open sourced by Google Research [24](https://github.com/google-research/bert]). Many NLU tasks can be retrained on custom data in short time using pretrained Bert models. BERT NLU application helpers are available on Hugging face for many different tasks, such as BerForSentencePrediction[21](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForNextSentencePrediction), BertForQuestionAnswering[22](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForQuestionAnswering), BertForMultipleChoice[23](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMultipleChoice), and many more.


###Problems encountered
------------
1. Custom Training time is very long. The amazon MARC dataset is very big. I have to use shard methos to select 1/10 of the dataset to train the model. I am running with CoLab Pro+ account, using Standard GPU and High RAM options. It took me about 2:58 hours to train the XML RoBerta based model on 1/10 of Amazon MARC dataset to sentimental classification.
2. The space requirement is high. I got "CUDA out of memory" when I train the XML-RoBertaa based model. I have to limit the batch size to overcome this problem. Similiary commands exists in Windows or Linux.
"!export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512' "
3. Hyperparameter tunning requires even more memory. I got "Cuda out of memory" again when searching for hyperparameter using the 1/10 shard of dataset. I have to shard the dataset by another 1/100 (totally sharded 1/1000, using only about 12,000 rows of training data) in order to run parameter search. Google research [23] [summarized](https://drive.google.com/file/d/1-XvHep0EITROHEFYKtQqv14dBRYKMBkZ/view?usp=sharing) the memory useg factors are:
* max_seq_length
* train_batch_size
* Model type: Bert_Base vs Bert_Large
* Optimzer

4. Hugging face token missing when I tried to push the trained model to hugging face automatically from CoLab. I fixed this problem by importing hugging face library, login interface and obtaining token. I learned a "fast paste" trick to make hugging face accept my token, quite fun.
5. CoLab runtime get disconnected after 90 sec after no activity. Turning off screen locker does not help, quite annoying. I upgraded to Colab Pro +, which is helpful. I am able to re-connect to the runtime after being discounted. My training job was not inturrupted by the disconnect.

   

###References
------------
1.   Ashish Vaswani, Noam Shazeer, Niki Parmar, et. al. Transformer Architecture - Attention is all you need [6/12/2017 https://arxiv.org/pdf/1706.03762.pdf]
2.   Jacob Devlin, Ming-Wei Chang, Kenton Lee, et. al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [10/11/2018  https://arxiv.org/abs/1810.04805v2]
3.   Bert-based-multilingual-uncased-sentimental model (by nlptown)[Ref https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment]
4.   xlm-roberta-base-finetuned-marc (by lewtun)[https://huggingface.co/lewtun/xlm-roberta-base-finetuned-marc]
5.   huggingface.io
6.   Aspect Based BERT[Ref https://arxiv.org/abs/2001.11316]
7.   Fine-grain BERT [https://ieeexplore.ieee.org/document/8947435]
8.   Joint intention BERT [https://arxiv.org/pdf/1902.10909.pdf]
9.   Emotion BERT [Ref https://www.researchgate.net/publication/350591267_Emotion_and_sentiment_analysis_of_tweets_using_BERT]
10.   Emotion BERT comparison [https://ieeexplore.ieee.org/abstract/document/9317379]
12.  RoBerta [https://arxiv.org/abs/1911.02116]
13.  Bert, RoBERTa, DistilBert, XLBert - which one to use. [https://towardsdatascience.com/bert-roberta-distilbert-xlnet-which-one-to-use-3d5ab82ba5f8]
14.  DistilBert [https://arxiv.org/abs/1910.01108]
15.  TinyBert [https://dair.ai/TinyBERT-Size_does_matter,_but_how_you_train_it_can_be_more_important/]
16.  MiniML [https://arxiv.org/abs/2002.10957]
17.  Smart Dynamic Padding [https://towardsdatascience.com/multilingual-amazon-reviews-classification-f55f8a650c9a]
18.  Tong's XML_RoBERTA based sentiment Amazon MARC [https://huggingface.co/tlttl/tluo_xml_roberta_base_amazon_review_sentiment_v3?text=I+like+you.+I+love+you]
19. Bahdanau, et. al. NMT with "Allign and Translate" [2014 https://arxiv.org/abs/1409.0473 ]
20. Minh-Thang, et. al. Attention based NMT [2015 https://arxiv.org/abs/1508.04025]
21. BertForSentencePrediction [https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForNextSentencePrediction]
22. BertForQuestionAnswering [https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForQuestionAnswering]
23. BertForMultipleChoice [https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMultipleChoice]
24. Bert Source code, by Google Research[https://github.com/google-research/bert]
25. Bert Out of memory issues [https://github.com/google-research/bert#out-of-memory-issues]
26. Keung, Phillip et. al. The Multilingual Amazon Reviews Corpus[https://arxiv.org/abs/2010.02573]
