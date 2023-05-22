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

interface TranscribedContent extends Content {
    transcript: String;
}

export interface WWDCTranscriptLoaderOptions {
    transcriptsUrl?: string;
    contentsUrl?: string;
}

export class WWDCTranscriptLoader
    extends BaseDocumentLoader
{
    private readonly transcriptsUrl: string;
    private readonly contentsUrl: string;

    constructor({
        transcriptsUrl = 'https://devimages-cdn.apple.com/wwdc-services/rd7a2338/9A892547-1713-4B1B-9D45-2C7A885BEDEF/transcript-digest-eng.json',
        contentsUrl = 'https://devimages-cdn.apple.com/wwdc-services/rd7a2338/9A892547-1713-4B1B-9D45-2C7A885BEDEF/contents.json',
    }: WWDCTranscriptLoaderOptions = {}) {
        super();
        this.transcriptsUrl = transcriptsUrl;
        this.contentsUrl = contentsUrl;
    }

    public async load(): Promise<Document[]> {
        const documents = await this.processSessions();
        return documents;
    }

    private async processSessions(): Promise<Document[]> {
        const documents: Document[] = [];
        const contents = await this.fetch_all_contents();
        const transcripts = await this.fetch_all_transcripts();

        for (const transcript of transcripts.slice(0, 1)) {
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
        for (const line of transcript.transcript) {
            let [time, text] = line;

            // remove text between the two ♪ characters and assign it to text
            text = text.replace(/♪.*♪/, '');

            // remove ♪ characters
            text = text.replace(/♪/g, '');

            // replace newline with space
            text = text.replace(/\n/g, ' ');

            // append text to result
            result += `${text}`;
        }

        // post-process result
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

        const response = await fetch(this.transcriptsUrl);
        const json = await response.json();

        for (const key in json) {
            const transcript = json[key] as Transcript;
            transcript.id = key;
            transcripts.push(transcript);
        }
 
        return transcripts
    }
}