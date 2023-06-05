import type { NextApiRequest, NextApiResponse } from 'next';
import { WWDCTranscriptLoader } from "@/utils/wwdcTranscriptLoader";
import { makeSummaryChain } from '@/utils/makechain';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import NextCors from 'nextjs-cors';
import { supabase } from '@/utils/supabase';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  await NextCors(req, res, {
    // Options
    methods: ['GET', 'HEAD', 'PUT', 'PATCH', 'POST', 'DELETE'],
    origin: '*',
    optionsSuccessStatus: 200, // some legacy browsers (IE11, various SmartTVs) choke on 204
  });
  
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
    const summary = await handleSummaryRequest(sessionId);
    console.log('summary: ', summary);
    res.status(200).json({ summary });
  } catch (error: any) {
    console.log('error', error);
    res.status(500).json({ error: error.message || 'Something went wrong' });
  }
};

async function handleSummaryRequest(sessionId: string): Promise<string> {
  const summary = await getSummaryFromDb(sessionId);
  if (summary) {
    return summary;
  }

  const newSummary = await generateSummary(sessionId);
  await writeSummaryToDb(sessionId, newSummary);
  return newSummary;
}

async function generateSummary(sessionId: string) {
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

  return response.text;
}

async function writeSummaryToDb(sessionId: string, summary: string) {
  const { data, error } = await supabase
    .from('session_summary')
    .insert([{ session_id: sessionId, summary: summary }]);

  if (error) {
    throw error;
  }

  return data;
}

async function getSummaryFromDb(sessionId: string): Promise<any> {
  const { data, error } = await supabase
    .from('session_summary')
    .select('summary')
    .eq('session_id', sessionId)
    .maybeSingle();

  if (error) {
    throw error;
  }

  if (!data) {
    return null;
  }

  return data.summary;
}