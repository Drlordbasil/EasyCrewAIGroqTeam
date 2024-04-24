import os
import subprocess
from typing import List, Callable
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from crewai_tools import tool, WebsiteSearchTool, ScrapeWebsiteTool, RagTool

def chunk_data(data: str, chunk_size: int = 4096) -> List[str]:
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

def process_chunks(chunks: List[str], process_func: Callable[[str], str]) -> str:
    return ''.join(process_func(chunk) for chunk in chunks)

def create_llm(model_name: str, groq_api_key: str) -> ChatGroq:
    return ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name=model_name
    )

@tool("File Manager")
def file_manager(command: str, file_path: str, content: str = "") -> str:
    """
    File management tool that supports various file operations.

    Args:
        command (str): The file operation command. Supported commands: 'read', 'write', 'delete', 'create_dir', 'list_dir'.
        file_path (str): The path to the file or directory.
        content (str, optional): The content to write to the file (only applicable for the 'write' command). Defaults to an empty string.

    Returns:
        str: The result of the file operation or an error message if an exception occurs.
    """
    try:
        if command == "read":
            with open(file_path, "r") as f:
                data = f.read()
                chunks = chunk_data(data)
                return process_chunks(chunks, lambda x: x)
        elif command == "write":
            chunks = chunk_data(content)
            with open(file_path, "w") as f:
                f.writelines(chunks)
            return f"File {file_path} written successfully."
        elif command == "delete":
            os.remove(file_path)
            return f"File {file_path} deleted successfully."
        elif command == "create_dir":
            os.makedirs(file_path, exist_ok=True)
            return f"Directory {file_path} created successfully."
        elif command == "list_dir":
            files = os.listdir(file_path)
            chunked_files = chunk_data(', '.join(files))
            return process_chunks(chunked_files, lambda x: f"Files in {file_path}: {x}")
        else:
            return "Invalid command. Supported commands: 'read', 'write', 'delete', 'create_dir', 'list_dir'."
    except FileNotFoundError:
        return f"File or directory '{file_path}' not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"

@tool("Code Runner")
def code_runner(file_path: str) -> str:
    """
    Code execution tool that runs a Python script file.

    Args:
        file_path (str): The path to the Python script file.

    Returns:
        str: The output of the code execution, including any errors or exceptions.
    """
    try:
        result = subprocess.run(["python", file_path], capture_output=True, text=True)
        if result.returncode == 0:
            return f"Code execution successful. Output:\n{result.stdout}"
        else:
            return f"Code execution failed. Error:\n{result.stderr}"
    except FileNotFoundError:
        return f"File '{file_path}' not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def create_agent(role: str, goal: str, backstory: str, llm: ChatGroq, tools: List[Callable]) -> Agent:
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )

def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")

    models = {
        "1": ("llama3-8b-8192", "LLaMA3 8b, Context Window: 8192 tokens"),
        "2": ("llama3-70b-8192", "LLaMA3 70b, Context Window: 8192 tokens"),
        "3": ("llama2-70b-4096", "LLaMA2 70b, Context Window: 4096 tokens"),
        "4": ("mixtral-8x7b-32768", "Mixtral 8x7b, Context Window: 32768 tokens"),
        "5": ("gemma-7b-it", "Gemma 7b, Context Window: 8192 tokens")
    }

    print("Choose a model:")
    print("\n".join(f"{key}: {description}" for key, (_, description) in models.items()))

    model_choice = input("Enter your choice (number): ")
    model_name = models.get(model_choice, ("llama3-8b-8192", "Default model LLaMA3 8b"))[0]

    llm = create_llm(model_name, groq_api_key)

    website_search_tool = WebsiteSearchTool()
    scrape_website_tool = ScrapeWebsiteTool()
    rag_tool = RagTool()

    agents = [
        create_agent('Researcher',
                     "As a Researcher, your primary goal is to gather, analyze, and synthesize information relevant to the project. Conduct thorough research to understand the problem domain, identify key factors influencing customer churn, and explore existing solutions. Your role is crucial in providing the team with the necessary background knowledge and insights to make informed decisions throughout the project lifecycle.",
                     "With your strong analytical skills and attention to detail, you excel at extracting meaningful insights from vast amounts of data. Your ability to identify patterns, trends, and correlations helps the team gain a deeper understanding of customer behavior and the factors driving churn. You collaborate closely with the Code Implementer and Human Interaction Specialist to ensure that the research findings are effectively translated into actionable strategies and solutions.",
                     llm, [file_manager, website_search_tool, scrape_website_tool, rag_tool]),
        create_agent('Code Implementer',
                     "As a Code Implementer, your primary responsibility is to translate the research findings and design specifications into functional and efficient code. You possess a deep understanding of programming languages, frameworks, and best practices required to build robust and scalable software solutions. Your role is essential in bringing the project to life by developing high-quality code that accurately reflects the intended functionality and performance.",
                     "With your strong problem-solving skills and attention to detail, you excel at breaking down complex requirements into manageable tasks and implementing them with precision. You collaborate closely with the Researcher to ensure that the code aligns with the project's objectives and incorporates the latest research insights. Your ability to write clean, maintainable, and well-documented code is crucial for the long-term success and maintainability of the project.",
                     llm, [file_manager, code_runner, rag_tool]),
        create_agent('Human Interaction Specialist',
                     "As a Human Interaction Specialist, your primary goal is to ensure that the project effectively addresses the needs and expectations of the end-users. You possess a deep understanding of human behavior, user experience design, and effective communication strategies. Your role is crucial in bridging the gap between the technical aspects of the project and the human factors that influence its success.",
                     "With your strong empathy and communication skills, you excel at understanding user requirements, gathering feedback, and translating them into actionable insights for the team. You collaborate closely with the Researcher and Code Implementer to ensure that the project delivers a user-centric solution that is intuitive, engaging, and delivers value to the target audience. Your ability to anticipate user needs, design compelling user interfaces, and create effective documentation is essential for the project's adoption and success.",
                     llm, [file_manager, rag_tool])
    ]

    user_question = "Create a project that predicts customer churn for a telecommunication company based on historical data. The model should identify key factors influencing churn and provide actionable insights to reduce customer attrition."

    tasks = [
        Task(
            description=f"As the Researcher, your task is to conduct a thorough analysis of the customer churn problem in the telecommunications industry. Review existing literature, case studies, and industry reports to identify the key factors that contribute to customer churn. Gather and synthesize relevant data from various sources to support the development of an accurate and reliable churn prediction model. Specifically:\n\n1. Investigate the common reasons why customers churn in the telecommunications industry, such as price sensitivity, service quality, competition, and customer demographics.\n2. Analyze historical customer data to identify patterns and correlations between different variables and churn behavior.\n3. Explore existing churn prediction models and techniques used in the industry to understand their strengths and limitations.\n4. Identify potential data sources, such as customer transactions, service usage logs, and customer feedback, that can be used to train and validate the churn prediction model.\n5. Summarize your findings in a comprehensive research report that provides a solid foundation for the project team to build upon.",
            agent=agents[0],
            expected_output="A comprehensive research report that summarizes the key factors influencing customer churn in the telecommunications industry, along with an analysis of historical customer data, existing churn prediction models, and potential data sources. The report should provide a solid foundation for the project team to develop an accurate and reliable churn prediction model."
        ),
        Task(
            description="As the Code Implementer, your task is to develop a machine learning model that accurately predicts customer churn based on the research findings and data provided by the Researcher. Utilize your programming skills and knowledge of machine learning frameworks to build a robust and efficient solution. Specifically:\n\n1. Review the research report and collaborate with the Researcher to understand the key factors influencing customer churn and the available data sources.\n2. Preprocess and transform the raw data into a suitable format for training the machine learning model. Handle missing values, outliers, and perform necessary feature engineering.\n3. Split the data into training, validation, and testing sets to facilitate model development and evaluation.\n4. Experiment with different machine learning algorithms, such as logistic regression, decision trees, random forests, or gradient boosting, to identify the most suitable approach for churn prediction.\n5. Train and fine-tune the selected model using the training data, utilizing techniques such as cross-validation and hyperparameter optimization to improve performance.\n6. Evaluate the trained model's performance using appropriate metrics, such as accuracy, precision, recall, and F1 score, on the validation and testing sets.\n7. Implement the final model in a production-ready format, ensuring efficient inference and seamless integration with the overall solution architecture.\n8. Document the model development process, including the chosen algorithm, feature importance, hyperparameters, and evaluation results, to facilitate maintainability and future enhancements.",
            agent=agents[1],
            expected_output="A fully implemented and trained machine learning model for customer churn prediction, along with the necessary data preprocessing and feature engineering steps. The model should be evaluated using appropriate metrics and documented for maintainability and future enhancements. The code should be production-ready and easily integrable with the overall solution architecture."
        ),
        Task(
            description="As the Human Interaction Specialist, your task is to ensure that the customer churn prediction project effectively addresses the needs and expectations of the end-users, which include the telecommunications company's decision-makers and customer retention teams. Collaborate with the Researcher and Code Implementer to design user-friendly interfaces and actionable insights that enable stakeholders to leverage the churn prediction model effectively. Specifically:\n\n1. Review the research findings and the developed churn prediction model to understand the key factors influencing customer churn and the model's capabilities.\n2. Engage with the telecommunications company's stakeholders to gather their requirements, pain points, and expectations regarding customer churn prediction and retention strategies.\n3. Design intuitive and visually appealing user interfaces that allow stakeholders to interact with the churn prediction model easily. Consider dashboards, reports, and data visualization techniques to present insights effectively.\n4. Develop clear and concise documentation and user guides that explain how to interpret the model's predictions, understand the key churn drivers, and take appropriate actions to retain at-risk customers.\n5. Collaborate with the Code Implementer to ensure that the user interfaces and insights are seamlessly integrated with the overall solution architecture and aligned with the stakeholders' needs.\n6. Conduct user testing and gather feedback to iteratively refine the user experience and ensure that the solution meets the stakeholders' expectations.\n7. Provide training and support to the stakeholders to facilitate the adoption and effective utilization of the churn prediction model in their day-to-day operations.\n8. Continuously monitor and gather user feedback post-deployment to identify opportunities for improvement and enhancements.",
            agent=agents[2],
            expected_output="User-friendly interfaces, actionable insights, and comprehensive documentation that enable the telecommunications company's stakeholders to effectively leverage the churn prediction model for customer retention. The Human Interaction Specialist should ensure that the solution meets the stakeholders' needs, is easily adoptable, and provides clear guidance on interpreting and acting upon the model's predictions."
        )
    ]

    crew = Crew(agents=agents, tasks=tasks, verbose=2)
    result = crew.kickoff()

    result_chunks = chunk_data(result)
    processed_result = process_chunks(result_chunks, lambda x: f"Chunk: {x}\n")

    print("Crew execution results:")
    print(processed_result)

if __name__ == "__main__":
    main()
