import type { NextApiRequest, NextApiResponse } from 'next';
import { WWDCTranscriptLoader } from "@/utils/wwdcTranscriptLoader";
import { makeSummaryChain } from '@/utils/makechain';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";


export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  // load sessionId from param
  const { sessionId } = req.query;

  if (Array.isArray(sessionId)) {
    res.status(400).json({ error: 'sessionId must be a string' });
    return;
  }

  if (sessionId == undefined) {
    res.status(400).json({ error: 'sessionId is required' });
    return;
  }

  console.log('sessionId', sessionId);
  try {
    const loader = new WWDCTranscriptLoader({ sessionId: sessionId });
    const documents = await loader.load();
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 4096,
      chunkOverlap: 400,
    });
    const splitDocs = await splitter.splitDocuments(documents);
    const chain = await makeSummaryChain(sessionId)
    const response = await chain.call({
      input_documents: splitDocs,
    });

    res.status(200).json(response);
  } catch (error: any) {
    console.log('error', error);
    res.status(500).json({ error: error.message || 'Something went wrong' });
  }
};