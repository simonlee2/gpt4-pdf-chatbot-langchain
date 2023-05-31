
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
Your mission is to write an outline of a video using its subtitles.
The format of the subtitles will be "[timestamp in seconds]: [subtitle]".
For each section and each key point, you should provide a timestamp to the original video section that this sentence is based on.
For example, a summary of a video section that starts at second 31 will be: "[31]: summary".

The subtitles are given between the triple quotes below:
"""
{text}
"""

Keep the response under 4000 tokens!
Your outline:
`;

const SECTION_TITLES_PROMPT_TEMPLATE = new PromptTemplate({
    template: SECTION_TITLES_PROMPT,
    inputVariables: ["text"],
});

const SUMMARY_PROMP = `Your mission is to write a conscise summary of a video using its title and chapter summaries.
The format of the chapter summaries will be "[chapter timestamp in seconds]: chapter summary".
For example, a summary of a chapter that starts at second 31 will be: "[31]: summary".

The title of the video is: {video_title}
The chapter summaries are given between the triple quotes below:
"""
{text}
"""

Your concise video summary:
`;

const SUMMARY_PROMP_TEMPLATE = new PromptTemplate({
    template: SUMMARY_PROMP,
    inputVariables: ["video_title", "text"],
});

const loader = new WWDCTranscriptLoader({ sessionId: 'wwdc2021-10176' });
const documents = await loader.load();
console.log({ documents });
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 3072,
    chunkOverlap: 300,
});
const splitDocs = await splitter.splitDocuments(documents);
const llm = new OpenAI({
    temperature: 0,
    verbose: true,
});

const chain = loadSummarizationChain(llm, {
    type: 'map_reduce', 
    combineMapPrompt: SECTION_TITLES_PROMPT_TEMPLATE,
    combinePrompt: SECTION_TITLES_PROMPT_TEMPLATE
});

const res = await chain.call({
    input_documents: splitDocs.slice(0, 30),
});
console.log({ res });