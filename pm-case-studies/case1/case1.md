---
classoption: fleqn
pagestyle: empty
geometry: 'margin=2cm'
fontsize: 12pt
header-includes:
    - \usepackage{graphicx}
    - \usepackage{caption}
    - \usepackage{siunitx}
    - \usepackage{longtable}
    - \usepackage{enumitem}
    - \setlength\parskip{11pt}
---

# Case 1: Project Selection and Developing a Business Case

## Overview

In this case study, we will perform two separate tasks: project selection and developing a business case. In Task 1, we perform project selection after evaluating multiple projects using the weighted scoring model. Following this in Task 2, we develop a business case for the project that we select in Task 1.

## Task 1: Project Selection

In this task, we apply the weighted scoring model to evaluate four projects and determine which project to choose. The projects are as shown in Table 1.1.

\begin{center}
\begin{tabular}{ | c | p{0.85\linewidth} | } 
\hline
Project & Name \\ 
\hline
1 & FIT3161\_Ting\_Automatic Identification of Autism Spectrum Disorder from Brain Networks Using Graph Deep Learning \\ 
\hline
2 & FIT3161\_MeeChin\_Online Forum Analytics\_3 topics\\ 
\hline
3 & FIT3161\_Raphael\_When Machines learn Crypto \\ 
\hline
4 & FIT3161\_Mei Kuan\_Gender Bias AI \\ 
\hline
\end{tabular}
\end{center}

\begin{center}
\text{Table 1.1: The projects to evaluate}
\end{center}

We will apply the following criteria to evaluate the projects in Table 1.1.

\begin{center}
\begin{tabular}{ | p{0.2\linewidth} | p{0.76\linewidth} | } 
\hline
Criteria & Description \\ 
\hline
Knowledge of topic & Knowledge of topic refers to the familiarity and awareness of the concepts associated with the project topic. A higher level of knowledge indicates a greater level of understanding of the fields related to the topic. \\ 
\hline
Social impact & Social impact refers to the change to address a pressing social challenge. A higher social impact indicates that the project can lead to a more positive change to society. \\ 
\hline
Technical ease & Technical ease refers to the level of easiness for the technical aspect of the project. A higher technical ease indicates that the project is easier to be implemented. \\ 
\hline
Project flexibility & Project flexibility refers to the degree of freedom that the project team has in determining the direction of the project. A higher level of project flexibility indicates that the project is more open-ended, allowing more room for exploration. \\ 
\hline
\end{tabular}
\end{center}

\begin{center}
\text{Table 1.2: The criteria used to evaluate the projects}
\end{center}

\newpage

Using the criteria shown in Table 1.2, we have evaluated the projects in Table 1.1 as shown in Table 1.3.

\begin{center}
\begin{tabular}{ | l | c | c | c | c | } 
\hline
Criteria & Project 1 & Project 2 & Project 3 & Project 4 \\ 
\hline
Knowledge of topic & Low & Medium & Low & High \\ 
\hline
Social impact & High & Low & Medium & Very high \\ 
\hline
Technical ease & Medium & High & Very low & Medium \\ 
\hline
Project flexibility & Low & High & Medium & High \\ 
\hline
\end{tabular}
\end{center}

\begin{center}
\text{Table 1.3: The evaluation of each project}
\end{center}

###### 

We will now design a scoring system for each criterion.

### Scoring system

For each criteria, we will assign a score as shown in Table 1.4:

\begin{center}
\begin{tabular}{ | c | c | } 
\hline
Rating & Score \\ 
\hline
Very high & 10.0 \\ 
\hline
High & 7.5 \\ 
\hline
Medium & 5 \\ 
\hline
Low & 2.5 \\ 
\hline
Very low & 0.0 \\ 
\hline
\end{tabular}
\end{center}

\begin{center}
\text{Table 1.4: The scoring system}
\end{center}

This scoring system will be consistently applied across all criteria.

This leads to the weighted scoring model as shown in Table 1.5.

##### 

\begin{center}
\textbf{Weighted Scoring Model for Gender Bias AI}
\end{center}

Prepared by: **Yap** Jin Heng, **Goh** Kang Qi, **Choo** Kah Poh Alika

Date: 16 April 2021

\begin{center}
\begin{tabular}{ | l | c | c | c | c | c | } 
\hline
Criteria & Weight & Project 1 & Project 2 & Project 3 & Project 4 \\ 
\hline
Knowledge of topic & 35\% & 2.50 & 5.00 & 2.50 & 7.50 \\ 
\hline
Social impact & 25\% & 7.50 & 2.50 & 5.00 & 10.00 \\ 
\hline
Technical ease & 20\% & 5.00 & 7.50 & 0.00 & 5.00 \\ 
\hline
Project flexibility & 20\% & 2.50 & 7.50 & 5.00 & 7.50 \\ 
\hline
\bfseries Weighted Project Scores & \bfseries 100\% & \bfseries 4.25 & \bfseries 5.38 & \bfseries 3.13 & \bfseries 7.63 \\ 
\hline
\end{tabular}
\end{center}

\begin{center}
\text{Table 1.5: The weighted scoring model}
\end{center}

\newpage

The scores for each project can be better visualised in Figure 1.1.

\begin{center}
\begin{figure}[h!]
  \includegraphics[width=\linewidth]{graph.png}
  \caption*{Figure 1.1: A graph showing the scores for the weighted scoring model}
  \label{fig:graph}
\end{figure}
\end{center}

The justification for our weighted scoring model is as explained below.

### Criteria

### Knowledge of topic

Project 1 is a topic on which our team has a low level of knowledge. We have insufficient knowledge of the field of neurology and psychology. This lack of knowledge restricts our ability to understand and detect the signs of autism from fMRI (functional magnetic resonance imaging) scans. As a result, we are unable to explore this topic on a deeper level.

Project 2 is a topic on which our team has a medium level of knowledge. We are all experienced in using online learning tools, particularly due to COVID-19 after the transition of many institutions to online education. Our personal experience gives us significant insights into online discussion forums in higher education, which is the focus of this project.

Project 3 is a topic on which our team has a low level of knowledge. This is especially true due to our lack of exposure to cryptography and steganography. This causes us to lack conceptual and technical awareness in these fields. These limitations will greatly impede our progress in this project.

Project 4 is a topic on which our team has a high level of knowledge. This is because gender issues such as bias and inequality have been trending on popular media in recent times. Besides, our team has experience with computer vision due to its increasing popularity in many industries. Our greater understanding of this topic will ensure the project’s success.

### Social impact

Project 1 has a high social impact. This is due to its potential impact on the neurodiverse community. Autism is a problem that deserves great attention. This project will make a significant difference in identifying individuals that belong to the autistic community. The findings from this project may also lead to new and better ways for the neurotypical and autistic communities to communicate and coexist.

Project 2 has a low social impact. This project is targeted only towards individuals involved in online learning, which is not universally adopted. Although COVID-19 has improved the adoption of online learning in higher education, many students still have an overwhelming preference for face-to-face education and shy away from online forums. This project thus has a limited impact on society at large.

Project 3 has a medium social impact. Although cryptography and steganography are designed to be invisible to the user, applications of cryptography and steganography are highly widespread. Given the use of machine learning techniques, this project may be able to uncover vulnerabilities of common encryption and steganographic schemes, which will lead to more secure and invisible forms of encryption and steganography.

Project 4 has a very high social impact. Over the course of history, gender inequality has been a social problem in many communities. The field of AI is no exception to gender bias and this project has great potential to address this issue. The metamorphic relations studied in this project may lead to a better understanding of the perceptions of gender in social sciences. This project will lead society to a better tomorrow.

### Technical ease

Project 1 has a medium technical ease. This project has medium difficulty because we will be able to leverage existing scientific processing and deep learning libraries to visualise graphs and build neural networks. Since specialised tools for building neural networks are already popular, this project is not too difficult.

Project 2 has a high technical ease. This project is easy because we only have to apply technical skills in extracting data from online forums and analysing the data with existing natural language processing libraries. These skills can be easily learned online due to the wealth of educational resources targeted for beginners.

Project 3 has a very low technical ease. This project is very difficult because the project requires us to have a comprehensive understanding of cryptography and/or steganography in addition to machine learning. There are also fewer popular frameworks for encryption, leading to a possible need for us to develop our own encryption models. From a technical perspective, this is difficult to implement.

Project 4 has a medium technical ease. This project focuses on deep learning, where easy-to-use libraries already exist. However, we will need to study the metamorphic testing methodology and apply it to the model, which may cause some difficulties. Overall, this leads to medium technical ease.

### Project flexibility

Project 1 allows for a low project flexibility. This project is not very flexible because the direction of the project has been rigidly specified. The given neural network has to work exclusively on data obtained from the fMRI time series and the direction of the project study has been specified to focus on graph adjacency matrix construction. Since the input data and the direction of the study have been clearly specified, this gives us little flexibility to adapt the project according to our findings.

Project 2 allows for a high project flexibility. The only specification of this project is to focus on online discussion forums in higher education. No approach has been explicitly specified for the study to be conducted. Since there are no specific directions for the project to follow, we have great flexibility to determine the direction of the project.

Project 3 allows for a medium project flexibility. This project is flexible enough that the approach for machine learning is not rigidly specified. However, the direction of the study is constrained to cryptography and steganography, which do not allow much room for exploration. Due to the limitations in the reach of cryptography and steganography, this project has medium flexibility.

Project 4 allows for a high project flexibility. This project's main direction is to investigate gender bias, but the details of gender bias, including the definition of bias in AI, are left to the project team. No approach has been explicitly specified in the project topic specification. Hence, this project allows for a high degree of flexibility.

### Weightage of criteria

Although all criteria mentioned make a significant difference to the project's success, some criteria are more important than others. In the order of most important to least important, the project criteria and their corresponding weights are:

1. Knowledge of topic: 35%
2. Social impact: 25%
3. Technical ease: 20%
4. Project flexibility: 20%

Knowledge of topic is the most important criterion because our project team needs to understand the topic well enough before we choose it. If we lack knowledge about the concepts associated with the project topic, we would have to spend extensive time just studying these concepts. This is followed by social impact, where the project should be able to benefit the lives of the public. Social impact is very important because all projects should ultimately benefit society. Next in line is technical ease because all projects need to be technically feasible. A project with low technical ease has a high chance of failure. Project flexibility is equally important as technical ease because we wish to be able to adapt the direction of the project according to our findings and interests. However, project flexibility does not rank as high as knowledge of topic or social impact because our project team is willing to work on any topic, however rigidly specified it may be.

### Conclusion

Based on the criteria above, Project 4 is the recommended choice for our project. Project 4 has the highest weighted score out of all projects. Project 4 allows us to take advantage of our high degree of knowledge of the topic, in addition to its very high social impact, medium technical ease, and high project flexibility. In particular, Project 4 stands out as having the greatest social impact out of all four projects, which is the second-most important criterion in project selection. Thus, Project 4 is the best among the four projects.

\newpage

## Task 2: Developing a Business Case

Based on the project we selected in Task 1, we have developed the following business case.

# 

\begin{center}
\textbf{Business Case for Gender Bias AI}
\end{center}

Prepared by: **Yap** Jin Heng, **Goh** Kang Qi, **Choo** Kah Poh Alika

Date: 16 April 2021

###### 

\begin{longtable}{ | p{\linewidth} | } 
\hline
\bfseries{1.0 Introduction/Background} \\\\
We are a team of three Monash University students with a great passion for AI. Our team is interested and motivated to work on an AI project with social impact. We wish to work on a project that has positive social implications in addition to furthering the understanding of AI among researchers. Recent findings have shown the existence of gender bias in AI. Upon this discovery, we have found ourselves deeply concerned and driven to address this issue of gender bias in AI. \\\\
\hline
\bfseries{2.0 Business Objective} \\\\
We aim to improve existing AI, specifically face recognition systems, against gender bias. An additional aim is to investigate the relationship between AI accuracy and gender identification. This aligns with our goal to increase public trust in AI, with the ultimate aim of encouraging more organisations to adopt AI in their operations. By addressing gender bias, more fields such as security, healthcare, and financial services will be more confident to adopt AI to run their operations. This will lead to a more AI-integrated society. \\\\
\hline
\bfseries{3.0 Current Situation and Problem/Opportunity Statement} \\\\
Face recognition models that apply AI have long been adopted by the market. However, studies have proven that even state-of-the-art models have major flaws in gender identification. Current face recognition models such as Amazon Rekognition and Dlib show an average of 18\% higher misidentification rate in females compared to males (Collins, 2020). This poses major problems for industries such as government agencies and law enforcement, where they are more likely to misidentify female convicts and witnesses when adopting AI techniques. This compels us to investigate gender bias in AI to solve this problem. A more accurate AI face recognition model, which is less susceptible to gender bias, will encourage more organisations to adopt this technology in their daily operations. \\\\
\hline
\end{longtable}

\newpage

\begin{longtable}{ | p{\linewidth} | } 
\hline
\bfseries{4.0 Critical Assumption and Constraints} \\\\
The proposed AI face recognition model must be an improvement over existing face recognition models. This model must have higher accuracy in identifying faces, particularly in women, to reduce the disparity in model performance between different genders. In order to minimise project cost, the AI model must not require excessive computational resources. The project must develop this model to run on existing hardware and software so it can integrate seamlessly into everyday life. Upon release, the model must only require minimal technical support and provide an easily accessible Application Programming Interface (API) for developers. This is to encourage developers to utilise this face recognition model. \\\\
\hline
\bfseries{5.0 Analysis of Option and Recommendation} \\\\
There are three options for addressing this opportunity: \\
\begin{enumerate}
\item Do nothing. Existing state-of-the-art AI face recognition models are good enough for most purposes and there is little need to improve these models.
\item Modify an existing AI face recognition model and improve it against gender bias.
\item Develop our own face recognition model, specifically for the purpose of addressing gender bias.
\end{enumerate}
\\ Option 1 should first be discarded because statistics indicate existing models fail to perform adequately when identifying women compared to men. Option 2 is difficult to perform because we lack sufficient understanding regarding the architecture of existing face recognition models. Option 3 is the most viable as it provides our team with full control over the model. Therefore, after discussing with our project supervisor, option 3 is the preferred option. \\\\
\hline
\bfseries{6.0 Preliminary Project Requirements} \\\\
The project should develop the following: \\
\begin{enumerate}
\item An AI face recognition model on par with existing face recognition models with minimal performance disparity (< 5\%) in identifying males vs females. This model should accept images and videos with faces and be able to identify the gender and names of the subjects based on labelled samples.
\item A web platform to test the model with sample images and videos. This platform enables users to submit their images and videos to test the face recognition model.
\item A system to test, report, and visualise the performance of the AI face recognition system. This system will use metamorphic testing for evaluating model performance, which is a methodology to continuously compare new results of image classification with previous results. The project will need the system to report the results using attractive data visualisation techniques.
\end{enumerate}
\hspace{1cm}\\
\begin{enumerate}
\setcounter{enumi}{3}
\item Application Programming Interfaces (API) compatible with multiple programming languages and multiple operating systems for developers. The model will need to be accessible from popular programming languages for machine learning such as Python, Java, C++, R, and MATLAB. For accessibility, the model should be executable on popular operating systems such as Windows, Mac OS, Android, iOS, and various Linux distributions.
\item Documentation for the face recognition model for developers to reference. This documentation should be hosted on an independent domain. The documentation should be actively updated according to the latest changes to the software.
\item Other features suggested to the model if they add value and performance to the product.
\end{enumerate}
\\
\hline
\bfseries{7.0 Schedule Estimate} \\\\
The project supervisor would like to see the project completed within a year, allowing for some flexibility in the schedule. The AI face recognition model should remain relevant in the field for at least three years. \\\\
\hline
\bfseries{8.0 Potential Risks} \\\\
This project carries numerous risks. A major technical risk of this project is that the AI face recognition model might require more computational resources than expected. This may lead to it being incompatible with low-end devices. Another technical risk is that unexpected bugs in the system might cause the system to fail, which will in turn affect the performance of the product. These technical issues can all be mitigated with extensive software testing. \\\\
Besides that, another risk present in this project is that the time spent on developing this project may exceed our schedule estimate. Unexpected schedule changes may lead to delays in product development. Thus, in order to ensure that the project remains on schedule, our team should adopt the use of project management software such as Gantt charts and critical path analysis. \\\\
\hline
\end{longtable}

<!-- TODO: Check references -->
