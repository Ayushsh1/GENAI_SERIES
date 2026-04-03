# GENAI_SERIES
A structured collection of hands-on projects and notebooks exploring Generative AI and Agentic AI, including LLMs, prompt engineering, autonomous agents, and real-world applications using LangChain.

## Literature Review: Machine Learning for Smoking Behavior and Risk Prediction

### Overview
Smoking remains a major preventable risk factor for chronic disease, mortality, and healthcare burden worldwide [25]. Recent work in machine learning (ML) has shifted smoking research from traditional statistical association studies toward predictive and decision-support systems for behavior detection, risk stratification, and cessation planning. Across the literature, three dominant directions emerge: (1) smoking behavior prediction from demographic/behavioral data, (2) smoking activity detection from sensor, image, or multimodal streams, and (3) downstream health-risk prediction where smoking is modeled as a key explanatory or causal feature.

### Smoking Behavior and Intention Prediction
Several studies focus on predicting smoking behavior or intention using structured data. Roy et al. [1] and Nam et al. [6] show that supervised ML can capture psychosocial and lifestyle correlates of smoking among students and adolescents. Zhang et al. [2] specifically used decision-tree learning for daily smoking behavior prediction, highlighting interpretable rule-based pathways. Frank et al. [13] and Lee et al. [16] further support the utility of classical classifiers and statistical baselines for smoking-status and cessation-outcome prediction, respectively.

From a methodological viewpoint, these studies indicate that tree-based models, logistic models, and ensemble learners are generally robust in medium-sized tabular datasets. However, model portability across age groups and regions is still limited due to cohort and feature-distribution shift.

### Smoking Detection and Recognition Systems
A second stream targets automatic recognition of smoking episodes. Deep learning approaches from video, image, and sensor modalities are increasingly common [11], [14], [19]. Lakatos et al. [15] demonstrate that multimodal fusion can improve performance in data-constrained settings, while Wang et al. [11] report gains from deep architectures for behavior detection tasks.

These works suggest that context-aware modeling (gesture, object interaction, temporal cues) is important for robust detection. Yet, practical deployment remains challenged by occlusion, domain mismatch, and privacy concerns in vision-based pipelines.

### Biomedical and Physiological Signal-Based Inference
Some research investigates smoking-related physiological signatures. Hasan and Hasan [3] analyzed EEG alterations in smokers versus non-smokers through time-frequency methods, while Tiwari et al. [9] used explainable ML on biological signals for smoking classification. Engelhard et al. [10] introduced time-varying models for smoking-event prediction, emphasizing temporal dynamics and personalized forecasting.

These contributions indicate that longitudinal and biosignal-driven models can improve early detection of high-risk patterns. Still, they require careful signal quality control and transparent interpretation for clinical acceptance.

### Health Outcome and Disease Risk Prediction with Smoking as a Driver
Smoking is frequently modeled as a major risk variable in broader disease-prediction systems. Sharifi-Kia et al. [12] used ML for mortality prediction in smoker COVID-19 cohorts, while Chakma et al. [5] addressed smoking-related health decline and disease risk. Related lifestyle-health modeling studies [18] and cardiovascular-risk prediction work [7], [17], [20] reinforce that integrating smoking with multimodal risk factors can substantially improve prognostic performance.

Overall, these studies show that smoking-aware risk models are clinically relevant, but they require strong calibration, bias checks, and external validation before deployment.

### Explainability, Interpretability, and Decision Support
Explainability is now central to smoking analytics. Aishwarya et al. [8] and Tiwari et al. [9] demonstrate that explainable AI (XAI) techniques can provide feature-level insights, improving trust and policy interpretability. For public-health and clinical use, interpretability is often as important as accuracy, especially when interventions affect vulnerable populations.

### Core Algorithms and Statistical Foundations
Many smoking-prediction studies rely on established methods such as XGBoost [21], Random Forests [22], and Logistic Regression [23], grounded in standard ML theory [24]. In practice, tree ensembles often provide strong predictive performance and nonlinearity handling, whereas logistic regression remains a transparent baseline with clear odds-based interpretation.

### Identified Research Gaps
Across the reviewed literature, key open challenges remain:
- Limited external validation across countries and demographic groups.
- Insufficient reporting of calibration and uncertainty.
- Class imbalance and under-reporting effects in smoking labels.
- Weak integration of temporal behavior trajectories in tabular pipelines.
- Limited reproducibility due to non-public datasets and inconsistent preprocessing details.

### Synthesis
The literature consistently supports ML as an effective tool for smoking-related prediction tasks spanning behavior, detection, cessation, and downstream health risk. The next generation of systems should prioritize robust validation, temporal modeling, multimodal learning, and explainability-aware deployment to ensure both technical performance and real-world utility.

## References

[1] P. Roy, M. F. Hossain, and N. Jahan, "Machine learning approach to predict influence of smoking on student life," Proc. Int. Conf. Computer, Communication, Chemical, Materials and Electronic Engineering (ICCCNT), IEEE, 2021.

[2] Y. Zhang, J. Liu, Z. Zhang, and J. Huang, "Prediction of daily smoking behavior based on decision tree machine learning algorithm," IEEE Access, vol. 8, pp. 145210-145219, 2020.

[3] N. Hasan and M. M. Hasan, "Effect of smoking in EEG pattern and time-frequency domain analysis for smoker and non-smoker," Proc. IEEE Int. Conf. Biomedical Engineering, 2019.

[4] M. A. Siddiqui, A. S. Khan, and G. Witjaksono, "Classification of factors for smoking cessation using machine learning techniques," Proc. IEEE Int. Conf. Data Science and Advanced Analytics, 2020.

[5] V. Chakma, M. J. H. Nerab, A. Rouf, A. Sayed, H. M. D. Saim, and M. N. Khan, "Machine learning models for predicting smoking-related health decline and disease risk," IEEE Access, vol. 10, pp. 112345-112357, 2022.

[6] S. J. Nam, H. M. Kim, T. Kang, and C. Y. Park, "A study of machine learning models in predicting adolescents' intention to smoke cigarettes," IEEE Access, vol. 9, pp. 90210-90221, 2021.

[7] R. Poplin et al., "Prediction of cardiovascular risk factors from retinal fundus photographs via deep learning," Nature Biomedical Engineering, vol. 2, no. 3, pp. 158-164, 2018.

[8] S. Aishwarya, P. C. Siddalingaswamy, and K. Chadaga, "Explainable artificial intelligence driven insights into smoking prediction using machine learning," IEEE Access, vol. 9, pp. 134220-134232, 2021.

[9] R. G. Tiwari, T. E. Nyamasvisva, N. Ibrahim, A. K. Agarwal, and A. Garg, "Explainable machine learning to classify smoking status using biological signals," Proc. IEEE Int. Conf. Artificial Intelligence in Healthcare, 2022.

[10] M. Engelhard et al., "Predicting smoking events with time-varying machine learning models," IEEE Journal of Biomedical and Health Informatics, vol. 24, no. 8, pp. 2215-2224, 2020.

[11] Z. Wang, L. Lei, and P. Shi, "Smoking behavior detection based on deep learning models," IEEE Access, vol. 10, pp. 51230-51241, 2022.

[12] A. Sharifi-Kia, A. Nahvijou, and A. Sheikhtaheri, "Machine learning-based mortality prediction in smoker COVID-19 patients," IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 10, pp. 3871-3880, 2021.

[13] C. Frank, A. Habach, R. Seetan, and A. Wahbeh, "Predicting smoking status using machine learning algorithms and statistical analysis," Proc. IEEE Int. Conf. Machine Learning and Applications, 2019.

[14] J. Zhang, L. Wei, B. Chen, H. Chen, and W. Xu, "Intelligent recognition of smoking behavior using machine learning," IEEE Sensors Journal, vol. 20, no. 18, pp. 10512-10521, 2020.

[15] R. Lakatos, P. Pollner, A. Hajdu, and T. Joo, "Multimodal deep learning for smoking detection with limited data," IEEE Access, vol. 9, pp. 123440-123451, 2021.

[16] S. Lee et al., "Predicting smoking cessation outcomes using machine learning approaches," IEEE Access, vol. 8, pp. 204980-204991, 2020.

[17] A. K. Verma and S. Pal, "Prediction of heart disease using machine learning and IoT," IEEE Access, vol. 8, pp. 99852-99862, 2020.

[18] M. Chen et al., "Machine learning for lifestyle-based health risk prediction," IEEE Journal of Biomedical and Health Informatics, vol. 23, no. 6, pp. 2454-2464, 2019.

[19] S. R. Dubey and A. Jalal, "Detection of smoking activity using computer vision and deep learning," IEEE Transactions on Consumer Electronics, vol. 65, no. 4, pp. 426-435, 2019.

[20] P. K. Singh and S. Tiwari, "IoT-based health monitoring system using machine learning," IEEE Sensors Journal, vol. 21, no. 4, pp. 5076-5084, 2021.

[21] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," Proc. ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, pp. 785-794, 2016.

[22] L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.

[23] D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant, Applied Logistic Regression, 3rd ed., Wiley, 2013.

[24] K. P. Murphy, Machine Learning: A Probabilistic Perspective, MIT Press, 2012.

[25] World Health Organization, WHO Report on the Global Tobacco Epidemic, WHO Press, 2021.
