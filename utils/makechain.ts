import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain, LLMChain, MapReduceDocumentsChain, StuffDocumentsChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are an AI that can engage in a natural conversation about video transcripts from Apple's WWDC keynotes and sessions, 
providing answers in markdown format. Answer the following question based on the provided excerpt from a video transcript. 
If the answer cannot be found in the excerpt, please respond with "I'm not sure." 
If the question is not related to the given context, politely respond that you are tuned to only answer questions that are related to the context. 
Only include links that are referenced in the transcript; do not make up any links.

{context}

Question: {question}
Helpful answer in markdown:`;

export const makeQAChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-4', //change this to gpt-4 if you have access
  },
    { basePath: 'https://oai.hconeai.com/v1' }
  );

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};

import { loadSummarizationChain } from "langchain/chains";
import { PromptTemplate } from "langchain/prompts";

const SECTION_TITLES_PROMPT = `
Your mission is summarize a chunk of video subtitles into segments and provide a sentence to help reader fully understand the main point of the segment.
The format of the subtitles will be "[timestamp in seconds]: [subtitle]".
For each segment, you should provide a timestamp to the original subtitles that it's based on.
For example, a bullet point of a video segment that starts at second 31 will be: "[31]: summary".

The subtitles are given between the triple quotes below:
"""
{text}
"""

Your summary:
`;

const SECTION_TITLES_PROMPT_TEMPLATE = new PromptTemplate({
  template: SECTION_TITLES_PROMPT,
  inputVariables: ["text"],
});

const SUMMARY_PROMP = `Your mission is to write an outline of a video using a list of summaries based on segments of the video.
There can be multiple sections, sub-sections, and key points in each sub-section.
An example of a different outline is given between the triple slashes below:

///
# [9] Introduction
**Section Overview:** Steve and Tim introduce the new iPhone 99 and how its new features will change the world
## [12] Design
- [13] The new iPhone 99 is the most beautiful iPhone ever made
- [15] The exterior is made of a new material that is stronger than steel and lighter than aluminum
## [20] Camera
- [21] The new camera is the best camera ever made
- [22] It has a new sensor that is 10x more sensitive to light than the previous generation
///


The segment summaries are given between the triple quotes below:
"""
{text}
"""

Use the example above to format your outline for the video in Markdown:
`;

const SUMMARY_PROMP_TEMPLATE = new PromptTemplate({
  template: SUMMARY_PROMP,
  inputVariables: ["text"],
});

export const makeSummaryChain = async (sessionId: string) => {
  const mapLLM = new OpenAI({
    temperature: 0,
  });

  const combineLLM = new OpenAI({
    temperature: 0,
    modelName: 'gpt-4',
  });

  const returnIntermediateSteps = false;
  const verbose = true;

  const mapLLMChain = new LLMChain({ 
    prompt: SECTION_TITLES_PROMPT_TEMPLATE, 
    llm: mapLLM,
    verbose
  });
  const combineLLMChain = new LLMChain({
    prompt: SUMMARY_PROMP_TEMPLATE,
    llm: combineLLM,
    verbose,
  });
  const combineDocumentChain = new StuffDocumentsChain({
    llmChain: combineLLMChain,
    documentVariableName: "text",
    verbose,
  });

  return new MapReduceDocumentsChain({
    llmChain: mapLLMChain,
    combineDocumentChain,
    documentVariableName: "text",
    returnIntermediateSteps,
    verbose,
  });
};