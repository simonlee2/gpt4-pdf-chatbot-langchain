import { WWDCTranscriptLoader } from "./utils/wwdcTranscriptLoader";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAI } from "langchain/llms/openai";
import { LLMChain, loadSummarizationChain } from "langchain/chains";
import { PromptTemplate } from "langchain/prompts";


// main
const loader = new WWDCTranscriptLoader({ sessionId: 'wwdc2021-10176' });
const documents = await loader.load();
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1024,
    chunkOverlap: 100,
});
const splitDocs = await splitter.splitDocuments(documents);

const model = new OpenAI({
    temperature: 0,
});

const template = `I have a video transcript with timestamps in seconds and would like you to provide me with summaries and key points for each section with the following format:

# [timestamp](t=(timestamp)s) Section Title

**Section Overview:** A brief summary of the section.

## [timestamp](t=(timestamp)s) Sub-section Title

- [timestamp](t=(timestamp)s A highlight in sub-section
- [timestamp](t=(timestamp)s A highlight in sub-section

## [timestamp](t=(timestamp)s) Sub-section Title

- [timestamp](t=(timestamp)s A highlight in sub-section
- [timestamp](t=(timestamp)s A highlight in sub-section

Transcript:

"{text}"

Please provide the summaries and key points for each section.
`

const prompt = new PromptTemplate({
    template,
    inputVariables: ["text"],
})

let summary = "";
for (const doc of splitDocs.slice(0, 5)) {
    const simpleChain = new LLMChain({ llm: model, prompt });
    const result = await simpleChain.call({ text: doc.pageContent });
    summary += result.text;
}
console.log(summary);

// Map-reduce summary

// const chain = loadSummarizationChain(model, { combineMapPrompt: prompt, combinePrompt: prompt });
// const res = await chain.call({
//     input_documents: splitDocs.slice(0, 10),
// });
// console.log({ res });

// const docOutput = await splitter.splitDocuments(documents.slice(0, 1));

/*-------------------------------------------- V2 --------------------------------------------*/

