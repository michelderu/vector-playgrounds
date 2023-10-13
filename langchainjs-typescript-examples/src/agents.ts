  //Import the OpenAPI Large Language Model (you can import other models here eg. Cohere)
  import { OpenAI } from "langchain/llms/openai";

  //Import the agent executor module
  import { initializeAgentExecutor } from "langchain/agents";

  //Import the SerpAPI and Calculator tools
  import { SerpAPI } from "langchain/tools";
  import { Calculator } from "langchain/tools/calculator";

  //Load environment variables (populate process.env from .env file)
  import * as dotenv from "dotenv";
  dotenv.config();

  export const run = async () => {

      //Instantiante the OpenAI model 
      //Pass the "temperature" parameter which controls the RANDOMNESS of the model's output. A lower temperature will result in more predictable output, while a higher temperature will result in more random output. The temperature parameter is set between 0 and 1, with 0 being the most predictable and 1 being the most random
      const model = new OpenAI({ temperature: 0 });

      //Create a list of the instatiated tools
      const tools = [new SerpAPI(), new Calculator()];

      //Construct an agent from an LLM and a list of tools
      //"zero-shot-react-description" tells the agent to use the ReAct framework to determine which tool to use. The ReAct framework determines which tool to use based solely on the tool’s description. Any number of tools can be provided. This agent requires that a description is provided for each tool.
      const executor = await initializeAgentExecutor(
      tools,
      model,
      "zero-shot-react-description"
      );
      console.log("Loaded agent.");

      //Specify the prompt
      const input =
      "Who is Beyonce's husband?" +
      " What is his current age raised to the 0.23 power?";
      console.log(`Executing with input "${input}"...`);

      //Run the agent
      const result = await executor.call({ input });

      console.log(`Got output ${result.output}`);
  };

  run();
