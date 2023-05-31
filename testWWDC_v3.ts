
import { WWDCTranscriptLoader } from "./utils/wwdcTranscriptLoader";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAI } from "langchain/llms/openai";
import { loadSummarizationChain } from "langchain/chains";
import { PromptTemplate } from "langchain/prompts";

interface SectionSummary {
    title: string;
    summary: string;
    highlights: string[];
}

interface VideoInfo {
    title: string;
    url: string;
    id: string;
}

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

const loader = new WWDCTranscriptLoader({ sessionId: 'wwdc2021-10176' });
const documents = await loader.load();
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 4096,
    chunkOverlap: 400,
});
const splitDocs = await splitter.splitDocuments(documents);

const llm = new OpenAI({
    temperature: 0,
    modelName: "gpt-4",
});

const chain = loadSummarizationChain(llm, {
    type: 'map_reduce', 
    combineMapPrompt: SECTION_TITLES_PROMPT_TEMPLATE,
    combinePrompt: SUMMARY_PROMP_TEMPLATE,
    verbose: true,
});

const res = await chain.call({
    input_documents: splitDocs,
});
console.log({ res });