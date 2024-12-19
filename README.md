# Job-Matching-and-Job_Seekers-Chat-System

## Project Overview
This project comprises two main components:

Job Matching Script: Automates the process of matching resumes to job descriptions using various similarity measures (TF-IDF, Jaccard, BERT, Word2Vec). Provides insights into the relevance of resumes to job postings using visualization and interactive analysis.

Chat Bot for Job Seekers: A conversational AI tool designed to assist job seekers by providing quick insights, answering queries about job matches, and managing job-related conversations.

## Part 1: Job Matching Script
Description
This script analyzes and ranks resumes against job postings using advanced natural language processing (NLP) techniques. Multiple similarity measures are implemented to compare resumes with job descriptions. Visualizations, interactive plots, and tabular results are provided to help job seekers quickly identify the most suitable jobs.

### Features
#### Data Cleaning and Preprocessing:

Normalizes text (lowercasing, removing special characters, etc.).
Handles missing values to ensure clean and accurate results.

#### Similarity Calculation Methods:

TF-IDF: Measures cosine similarity between term-weighted vectors.
Jaccard Similarity: Calculates the overlap of unique terms between resumes and jobs.
BERT + FAISS: Leverages contextual embeddings from the BERT model and accelerates similarity search using FAISS.
Word2Vec: Captures semantic similarities using word embeddings.

### Visualization and Results Display:

#### Heatmaps for similarity comparisons.
![image](https://github.com/user-attachments/assets/3aedd53c-f646-4c74-86d0-133c5192d65b)

![image](https://github.com/user-attachments/assets/92c9f98b-a8ab-4130-a5ae-aade2d5e8655)


#### Bar plots for top terms in resumes and job descriptions.
![image](https://github.com/user-attachments/assets/0e23029d-0df2-4d58-b269-6149c15f5f03)

#### Interactive scatter plots to explore relationships between resumes and job matches.

![image](https://github.com/user-attachments/assets/107d73bb-d707-40e0-9aa8-67e4e8ab220e)

![image](https://github.com/user-attachments/assets/86ade0b5-e34e-4a72-8570-7c8800c902da)


#### Tabular summaries of top matches for each similarity method.

![Screenshot (15)](https://github.com/user-attachments/assets/a56c3795-eb20-4fb9-9dde-47d0508c1d70)



### Prerequisites
Libraries Required:

pandas, numpy, matplotlib, seaborn, plotly, nltk, faiss, gensim
Hugging Face Transformers (transformers), sklearn
Install missing packages using:
bash
Copy code
pip install pandas numpy matplotlib seaborn plotly transformers sklearn faiss gensim
Data Input:

Resumes (Res.csv): A CSV file containing resumes with an "ID" and "Resume_str" column.
Job Postings (pos.csv): A CSV file containing job descriptions with a "title" and "description" column.

### How to Use
Load Data:

Upload Res.csv and pos.csv in the appropriate format.
Run Similarity Calculations:

Choose a similarity method (e.g., TF-IDF, Jaccard).
Execute the corresponding function to compute similarity scores and display results.

### Key Takeaways
#### 1. TF-IDF Results
The TF-IDF algorithm appears to provide a meaningful ranking of resumes, highlighting roles such as "Administrative Assistant," "Marketing Coordinator," and "Project Architect."
Average Similarity scores are relatively low (ranging around 0.11–0.22), which is expected for sparse data like resumes.
TF-IDF is performing well in identifying matches where textual overlap (job titles and keywords) is higher.
#### 2. Jaccard Similarity Results
Jaccard scores are consistently low, with Average Similarity scores around 0.09–0.15. This reflects its limitation in comparing longer texts (e.g., resumes), as Jaccard is highly sensitive to small differences in word overlap.
Matches like "Administrative Assistant" and "Marketing Coordinator" appear frequently, indicating that resumes with these roles likely share a higher proportion of common words.
#### 3. BERT + FAISS Results
BERT embeddings with FAISS yield the most robust results, with Average Similarity scores in the 49–65 range, which is higher than other methods.
This method captures semantic relationships between resumes and roles, explaining why matches like "Administrative Coordinator" and "Marketing Coordinator" are consistent.
This algorithm is particularly effective in identifying resumes with related but not identical content, making it valuable for nuanced role matching.
#### 4. Word2Vec Results
The Word2Vec-based scores are notably high (around 0.99), suggesting that embeddings are overfitting on certain repetitive features or failing to differentiate subtle variations.
Matches like "SALES" and "Legal Secretary" dominate, which may not reflect meaningful semantic relationships but rather generic token similarities.
While Word2Vec can provide value, it appears less reliable for resume-to-job matching in this case.
### Recommended Algorithms
BERT + FAISS: The best-performing algorithm for capturing semantic and contextual relevance. It is ideal for finding resumes that align with nuanced job descriptions.
TF-IDF: Performs well in identifying straightforward textual overlaps and is computationally lightweight. Useful for bulk pre-screening.
Jaccard: Has limited utility due to its reliance on exact word overlap, which is less relevant for resume data.
Word2Vec: Its high scores likely indicate overgeneralization, making it less useful for distinguishing nuanced matches.


## Part 2: Chat Bot for Job Seekers
### Description
The ChatBot module leverages conversational AI to streamline communication between job seekers and HR. It allows job seekers to interact with the system to find relevant job openings, ask questions, and receive timely updates throughout the application process.

![image](https://github.com/user-attachments/assets/03a36db4-23b7-453f-8766-24c668ac10ea)

This section outlines how the fine-tuned conversational model (using GPT-Neo and a conversational dataset) enhances career advisory responses. This approach leverages a pre-trained model, fine-tuned on domain-specific data, to provide tailored and accurate suggestions.

### Conversational Dataset Design
The dataset is structured with questions and answers addressing diverse career-related topics, such as:
Data science job roles
Resume-building for IT and marketing
Tools and skills for specific professions
Certifications for construction and IT careers

### Why it works:

The conversational format aligns with real-world user queries.
It captures a range of questions that span technical and non-technical roles, improving generalization.
The explicit Q&A structure aids in building contextual understanding for diverse inquiries.

### Fine-Tuning Process
Using GPT-Neo (EleutherAI/gpt-neo-125M), the model was fine-tuned on the Q&A dataset with the following steps:

#### Dataset Preparation:
Combined the question and answer into a single text field for input-output mapping.
Tokenized with truncation and padding to ensure uniform input sequences.

#### Fine-Tuning Objectives:
Train the model to accurately map diverse questions to domain-specific answers.
Improve fluency, relevance, and consistency in generating career advice.

#### Training Strategy:
Optimized using Trainer from Hugging Face, with hyperparameters tailored for conversational tasks:
Batch size: 4 for efficient computation.
Epochs: 3 to balance accuracy and training time.
Learning rate: 5e-5 to refine model weights incrementally.
Evaluation on a 10% test split to monitor generalization.

#### Final Deployment:
Fine-tuned model and tokenizer saved for deployment.
Integrated with Gradio for a user-friendly interface, allowing real-time interaction.

###  Key Strengths of the Fine-Tuned Model
Domain-Specific Insights:
The model provides targeted advice for data science, IT, marketing, and other careers.
Fine-tuning enhances its ability to address niche queries like resume optimization and skill prioritization.

Improved Contextual Understanding:
By training on domain-specific Q&A, the model grasps the intent behind career-related questions.
For example, it differentiates between technical IT certifications (e.g., AWS) and general IT skills (e.g., troubleshooting).

Dynamic Interaction:
Gradio integration facilitates real-time question answering with adjustable parameters (e.g., response length, temperature).
It ensures the chatbot can handle diverse queries effectively.

### Opportunities for Further Optimization
Dataset Expansion:
Add more varied questions (e.g., specific industries, career paths for unique skills).
Include queries with ambiguous phrasing to test the model's contextual understanding.
Use RAG to utilize resume and job vector databases for more concise outputs.

Model Scaling:
Upgrade to larger versions of GPT-Neo or GPT-J for enhanced performance.
Explore newer architectures like LLaMA or Falcon for improved efficiency.

Interactive Features:
Integrate response feedback from users to fine-tune iteratively.
Add capabilities like generating sample resumes or recommending career resources.

Reducing overmatching:
TF-IDF or other vectorization methods to give higher weight to keywords (e.g., specific skills, job titles) to solve overmatching based on non-keywords such as conjunctions or other commonly used words.

Example queries:
"Show me the top matches for Resume ID 12345."
"Which jobs are best suited for this candidate?"
"Provide similarity scores for all positions related to 'Data Scientist'."


### This two-part project aims to simplify the hiring process by combining advanced NLP techniques for job matching with an intuitive conversational interface for Job seekers. By automating and enhancing candidate-job matching, this tool provides valuable insights, reduces manual effort, and improves hiring efficiency.

## Literature Review and Existing Research

# Risks and Mitigating Risks

When setting up machine learning systems for matching resumes with job postings, there are many risks we aim to avoid.
One significant risk, widely covered in literature and research, is bias in the algorithm. Every pretrained model carries the risk of reflecting biases present in the training data. This is particularly relevant in models like BERT, which depend heavily on their training data. Evidence suggests that such models may favor specific demographics or exclude non-traditional resumes from advancing in the hiring process. To mitigate this risk, human review remains one of the most effective strategies, as it helps address various risks. These tools are designed with humans in mind, and supporting human decision-making is the ultimate goal. Auditing and the implementation of systematic tools for review can further help mitigate bias. For example, IBM's Fairness 360 tool can flag potential biases in programs.
Another risk is an over-reliance on keywords and matching in the selection process. Traditional approaches like TF-IDF and Jaccard Similarity depend heavily on exact word matches. However, language is complex, and these systems often struggle with understanding semantics and similarities.
There is also the risk of misinterpreting the context of words and tokens. For instance, these systems often struggle with recognizing sarcasm or understanding nuanced elements like soft skills or transferable experiences. Such contexts require extensive understanding that many language models lack. To mitigate this issue, we can expand pre-trained embeddings to better capture synonyms and related concepts, such as equating "software engineer" and "developer."
Another major challenge is false positives and false negatives in matching. These systems are designed with selection or non-selection as their end goal, leading to situations where irrelevant resumes are flagged as matches (false positives) or qualified candidates are overlooked (false negatives). Adjusting threshold similarity scores is an effective way to minimize these mismatches.

# Existing Literature and Approaches

The literature review began by exploring the history of resume matching systems. Due to their commercial applications, systematic mathematical approaches for informing hiring processes have existed since at least the 1970s. Early models often excluded variables like demographic information. According to DeLaCruz et al., our tolerance for algorithmic mistakes should be lower than for human error.
Some of the earliest machine-learning approaches to resume matching were relatively simple, relying on basic string matching, keyword extraction, and statistical methods. However, these methods struggled significantly with semantics compared to modern programs (Bocharova and Malakhov).
To summarize the types of AI used in resume matching, TF-IDF and Jaccard Similarity are considered traditional methods. While not as sophisticated as newer approaches like BERT, they are easier to implement and faster. These methods are still used by smaller enterprises. Their primary weaknesses lie in their limited ability to understand semantics, such as synonyms and related concepts, due to their dependence on exact keyword matching. Job seekers can improve their chances of selection in systems using TF-IDF or Jaccard Similarity by tailoring their resumes to include keywords from job descriptions.
Among modern approaches, BERT is perhaps the most widely used. With its focus on semantic understanding, it achieves a balance between efficiency and accuracy. However, BERT is not immune to errors, and it requires more time and effort to integrate and set up effectively.
A hybrid approach, combining traditional and advanced methods, offers a middle ground. It captures semantic relationships and contextual understanding like BERT but is the most computationally expensive option and carries a risk of overfitting.

# Conclusion

In conclusion, BERT, supported by FAISS, proved to be the best-performing algorithm, outperforming Word2Vec, Jaccard Similarity, and TF-IDF in its ability to capture semantic and contextual relevance. We recommend using this approach for resume matching.

