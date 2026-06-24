/**
 * 最简单的 LangChain 调用示例（无 RAG）
 * LangSmith 通过 .env 中的 LANGCHAIN_TRACING_V2 等变量自动上报链路
 */
import "dotenv/config";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: process.env.OPENAI_BASE_URL },
  model: process.env.MODEL_NAME ?? "qwen-plus",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "你是一个友好的助手，用简洁的中文回答。请用鲁迅的语气和我对话。"],
  ["human", "{input}"],
]);

const chain = RunnableSequence.from([prompt, llm, new StringOutputParser()]);

export async function chat(input) {
  return chain.invoke(
    { input },
    {
      runName: "simple-chat",
      tags: ["demo"],
    },
  );
}

const question =
  process.argv.slice(2).join(" ") || "用一句话介绍 LangSmith 是做什么的";

const answer = await chat(question);
console.log(`问: ${question}`);
console.log(`答: ${answer}`);
console.log(
  `\nLangSmith 项目: ${process.env.LANGCHAIN_PROJECT ?? "default"}`,
);
