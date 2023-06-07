import { Document } from 'langchain/document';
import { BaseDocumentLoader } from 'langchain/document_loaders/base';

interface Content {
    id: string;
    eventId: string;
    webPermalink: string;
    description: string;
    title: string;
    platforms: string[];
    primaryTopicID: number;
    topicIds: number[];
    deliveryLanguage: string;
    related?: {
        activities?: string[];
        resources?: number[];
    };
}

interface Transcript {
    id: string;
    language: string;
    transcript: [time: number, text: string][];
}

interface Manifest {
    updated: string;
    digest: {
        etag: string;
        url: string;
    },
    individual: {
        [sessionId: string]: {
            etag: string;
            url: string;
        };
    };
}

export interface WWDCTranscriptLoaderOptions {
    sessionId?: string;
}

export class WWDCTranscriptLoader
    extends BaseDocumentLoader
{
    private readonly sessionId?: string;
    private readonly transcriptsManifestUrl: string = 'https://devimages-cdn.apple.com/wwdc-services/d73c6be4/58F4932B-A6C0-4997-A114-551E0C803B53/transcript-manifest-eng.json'
    private readonly transcriptsDigestUrl: string = 'https://devimages-cdn.apple.com/wwdc-services/d73c6be4/58F4932B-A6C0-4997-A114-551E0C803B53/transcript-digest-eng.json'
    private readonly contentsUrl: string = 'https://devimages-cdn.apple.com/wwdc-services/d73c6be4/58F4932B-A6C0-4997-A114-551E0C803B53/contents.json'

    constructor({
        sessionId,
    }: WWDCTranscriptLoaderOptions = {}) {
        super();

        this.sessionId = sessionId;
    }

    public async load(): Promise<Document[]> {
        if (this.sessionId) {
            return await this.processSession(this.sessionId);
        } else {
            return await this.processSessions();
        }
    }

    private async processSession(sessionId: string): Promise<Document[]> {
        const contents = await this.fetch_all_contents();
        const transcript = await this.fetch_transcript(sessionId);
        const id = transcript.id;
        const content =contents.get(id);
        const processed_transcript = this.process_transcript(transcript);

        if (content) {
            const document = new Document({
                pageContent: processed_transcript,
                metadata: {
                    id,
                    title: content.title,
                    description: content.description,
                    source: content.webPermalink,
                },
            }); 
            return [document];
        } else {
            return [];
        }
    }

    private async processSessions(): Promise<Document[]> {
        const documents: Document[] = [];
        const contents = await this.fetch_all_contents();
        const transcripts = await this.fetch_all_transcripts();

        for (const transcript of transcripts) {
            const id = transcript.id;
            const content = contents.get(id);
            const processed_transcript = this.process_transcript(transcript);

            if (content) {
                const document = new Document({
                    pageContent: processed_transcript,
                    metadata: {
                        id,
                        title: content.title,
                        description: content.description,
                        source: content.webPermalink,
                    },
                });
                documents.push(document);
            }
        }
        return documents;
    }

    private process_transcript(transcript: Transcript): string {
        let result = "";
        let include_timestamp = true;
        for (const line of transcript.transcript) {
            let [time, text] = line;

            // remove text between the two ♪ characters and assign it to text
            text = text.replace(/♪.*♪/, '');

            // remove ♪ characters
            text = text.replace(/♪/g, '');

            // append text to result
            if (include_timestamp) {
                result += `[${time}]: ${text}`;
            } else {
                result += text;
            }

            // if text ends with \n\n
            if (text.endsWith('\n\n')) {
                include_timestamp = true;
            } else {
                include_timestamp = false;
            }
        }

        // post-process result
        result = result.replace(/\n\n/g, ' ');
        // remove multiple spaces
        result = result.replace(/ +/g, ' ');
        // remove ' \.'
        result = result.replace(/ \./g, '.');

        return result;
    }

    private post_process_description(description: string): string {
        description = description.replace(/’/g, '\'');
        description = description.replace(/“/g, '\"');
        description = description.replace(/”/g, '\"');
        description = description.replace(/\n/g, ' ');
        description = description.replace(/ +/g, ' ');
        return description;
    }

    private async fetch_all_contents(): Promise<Map<string, Content>> {
        const contents: Map<string, Content> = new Map();
        
        const response = await fetch(this.contentsUrl);
        const json = await response.json();
        const contentsJson = json['contents'];
        for (const content of contentsJson) {
            content.description = this.post_process_description(content.description);
            contents.set(content.id, content);
        }

        return contents;
    }

    private async fetch_all_transcripts(): Promise<Transcript[]> {
        const transcripts: Transcript[] = [];

        const response = await fetch(this.transcriptsDigestUrl);
        const json = await response.json();

        for (const key in json) {
            const transcript = json[key] as Transcript;
            transcript.id = key;
            transcripts.push(transcript);
        }
 
        return transcripts
    }

    private async fetch_transcript(sessionId: string): Promise<Transcript> {
        const response = await fetch(this.transcriptsManifestUrl);
        const manifest = await response.json() as Manifest;
        const transcriptManifest = manifest.individual[sessionId];
        const transcriptManifestResponse = await fetch(transcriptManifest.url)
        const transcriptManifestJson = await transcriptManifestResponse.json();
        const transcript = transcriptManifestJson[sessionId] as Transcript;
        transcript.id = sessionId;
        return transcript;
    }
}