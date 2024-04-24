import os
import pandas as pd
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from crewai_tools import tool

def chunk_data(data, chunk_size=4096):
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

def process_chunks(chunks, process_func):
    results = []
    for chunk in chunks:
        result = process_func(chunk)
        results.append(result)
    return ''.join(results)

def main():
    # Environment setup
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY is not set in the environment variables.")
        return

    # Model selection details
    models = {
        "1": ("llama3-8b-8192", "LLaMA3 8b, Context Window: 8192 tokens"),
        "2": ("llama3-70b-8192", "LLaMA3 70b, Context Window: 8192 tokens"),
        "3": ("llama2-70b-4096", "LLaMA2 70b, Context Window: 4096 tokens"),
        "4": ("mixtral-8x7b-32768", "Mixtral 8x7b, Context Window: 32768 tokens"),
        "5": ("gemma-7b-it", "Gemma 7b, Context Window: 8192 tokens")
    }

    print("Choose a model:")
    for key, (_, description) in models.items():
        print(f"{key}: {description}")

    model_choice = input("Enter your choice (number): ")
    model = models.get(model_choice, ("llama3-8b-8192", "Default model LLaMA3 8b"))[0]

    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name=model
    )

    # Custom file management tool
    @tool("File Manager")
    def file_manager(command: str, file_path: str, content: str = "") -> str:
        """Useful for managing files and directories. Supports commands like 'read', 'write', 'delete', 'create_dir', 'list_dir'."""
        try:
            if command == "read":
                with open(file_path, "r") as f:
                    data = f.read()
                    chunks = chunk_data(data)
                    return process_chunks(chunks, lambda x: x)
            elif command == "write":
                chunks = chunk_data(content)
                with open(file_path, "w") as f:
                    for chunk in chunks:
                        f.write(chunk)
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

    # Redefining agents with business-oriented roles
    Business_Analyst = Agent(
        role='Business Analyst',
        goal="""Identify and clarify the business objectives behind the machine learning problem to ensure that the model development aligns with client needs and market demands.""",
        backstory="""You are adept at translating complex business needs into clear, actionable objectives for model development. Your insights drive the project's direction and ensure relevance to market and client demands.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[file_manager],
    )

    Data_Engineer = Agent(
        role='Data Engineer',
        goal="""Prepare and optimize data for effective model training, ensuring data quality and relevance to the specific business problem.""",
        backstory="""Specializing in data pipelines and preprocessing, you ensure that the data is ready and optimized for training, crucial for the success of the developed models.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[file_manager],
    )

    ML_Model_Trainer = Agent(
        role='ML Model Trainer',
        goal="""Configure and fine-tune machine learning models to best address the defined business problems, maximizing performance and efficiency.""",
        backstory="""As a machine learning expert, you fine-tune models to adapt to various business scenarios, ensuring the highest performance and relevance to the business objectives.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[file_manager],
    )

    Solution_Architect = Agent(
        role='Solution Architect',
        goal="""Design comprehensive machine learning-driven solutions that integrate seamlessly into client systems, providing clear competitive advantages.""",
        backstory="""With a broad understanding of both technology and business applications, you architect solutions that are innovative, scalable, and directly contribute to client success.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[file_manager],
    )

    Sales_Engineer = Agent(
        role='Sales Engineer',
        goal="""Utilize the insights and capabilities of developed models to create compelling proposals that clearly demonstrate the value to potential clients.""",
        backstory="""You bridge the gap between technical capabilities and client needs, crafting proposals that highlight the practical benefits of the machine learning solutions.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[file_manager],
    )

    # User interaction and task definitions
    user_question = "Create a project that predicts customer churn for a telecommunication company based on historical data. The model should identify key factors influencing churn and provide actionable insights to reduce customer attrition."

    task_business_analysis = Task(
        description=f"Identify the business goals and objectives for the ML project based on this input:\n\n{user_question}",
        agent=Business_Analyst,
        expected_output="A defined set of business goals and objectives for the ML project."
    )

    task_data_preparation = Task(
        description="Prepare and manage the data to be ready for model training, ensuring quality and alignment with the business problem. Create a '/data' directory to store all the necessary data files for the project.",
        agent=Data_Engineer,
        expected_output="Well-prepared and optimized data ready for model training stored in the '/data' directory."
    )

    task_model_training = Task(
        description="Fine-tune the machine learning model to suit the defined business objectives, ensuring optimal performance and relevance.",
        agent=ML_Model_Trainer,
        expected_output="A fine-tuned machine learning model ready for deployment."
    )

    task_solution_design = Task(
        description="Design a comprehensive solution that integrates the ML model into the client's operational framework, enhancing their business operations.",
        agent=Solution_Architect,
        expected_output="A detailed ML-driven solution architecture that can be readily implemented."
    )

    task_sales_proposal = Task(
        description="Create a detailed proposal that demonstrates the value of the ML model to potential clients, focusing on the benefits and competitive advantages.",
        agent=Sales_Engineer,
        expected_output="A compelling sales proposal that effectively communicates the benefits of the ML solution."
    )

    crew = Crew(
        agents=[Business_Analyst, Data_Engineer, ML_Model_Trainer, Solution_Architect, Sales_Engineer],
        tasks=[task_business_analysis, task_data_preparation, task_model_training, task_solution_design, task_sales_proposal],
        verbose=2
    )

    # Execute the crew
    result = crew.kickoff()
    
    # Chunk the result and process each chunk
    result_chunks = chunk_data(result)
    processed_result = process_chunks(result_chunks, lambda x: f"Chunk: {x}\n")
    
    print("Crew execution results:")
    print(processed_result)

if __name__ == "__main__":
    main()  # Run the main function
