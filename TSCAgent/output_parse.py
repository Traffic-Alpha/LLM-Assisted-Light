'''
@Author: WANG Maonan
@Date: 2023-09-18 17:52:04
@Description: Output parse
@LastEditTime: 2023-09-18 21:07:32
'''
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

class OutputParse(object):
    def __init__(self, env=None, llm=None) -> None:
        self.sce = env
        self.llm = llm

        self.response_schemas = [
            ResponseSchema(
                name="phase_id", description=f"output the id(int) of the traffic phase. For example, if the final decision is signal phase 1, please output 1 as a int."),
            ResponseSchema(
                name="explanation", description=f"Explain for the Crossing Guard why you make such decision.")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        self.format_instructions = self.output_parser.get_format_instructions()

    def parser_output(self, final_results:str) -> str:
        prompt_template = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    "Parse the problem response follow the format instruction.\nformat_instructions:{format_instructions}\n Response: {answer}")
            ],
            input_variables=["answer"],
            partial_variables={"format_instructions": self.format_instructions}
        )

        custom_message = prompt_template.format_messages(
            answer = final_results,
        )
        output = self.llm(custom_message)
        print(output)
        self.final_parsered_output = self.output_parser.parse(output.content)
        
        return self.final_parsered_output