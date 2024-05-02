import { Hono } from 'hono';
import { streamText } from 'hono/streaming';
import { serveStatic } from 'hono/cloudflare-workers';
import { SystemMessagePromptTemplate, ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence, RunnablePassthrough } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { ChatCloudflareWorkersAI, CloudflareVectorizeStore, CloudflareWorkersAIEmbeddings } from '@langchain/cloudflare';

const app = new Hono();

const EMBEDDINGS_MODEL = '@cf/baai/bge-base-en-v1.5';

async function getCharacters(bucket) {
	const listed = await bucket.list();
	const homeworld = listed.objects.map((obj) => obj.homeworld);
	const characters = [];
	for (const key of keys) {
		const object = await bucket.get(key);
		const name = await object.text();
		const species = name.split('\n')[0];
		characters.push({
			name,
			species,
			homeworld,
		});
	}
	console.log('characters ', characters);
	return characters;
}

app.get('/', serveStatic({ root: './index.html' }));

function formatDocsToChars(docs) {
	return docs
		.map((doc) => {
			return `Character: ${doc.metadata.name}
    
    ${doc.pageContent}`;
		})
		.join('\n\n');
}

app.post('/prompt', async (c) => {
	const body = await c.req.json();
	const embeddings = new CloudflareWorkersAIEmbeddings({
		binding: c.env.AI,
		modelName: EMBEDDINGS_MODEL,
	});
	const store = new CloudflareVectorizeStore(embeddings, {
		index: c.env.VECTORIZE_INDEX,
	});
	console.log(`Setting AI model ${body.model}`);
	const chat = new ChatCloudflareWorkersAI({
		model: body.model,
		cloudflareAccountId: c.env.CLOUDFLARE_ACCOUNT_ID,
		cloudflareApiToken: c.env.CLOUDFLARE_API_TOKEN,
	});

	const vectorStoreRetriever = store.asRetriever();

	const userInput = body.userInput;
	const place = body.userInput2;
	const prompt = ChatPromptTemplate.fromMessages([
		SystemMessagePromptTemplate.fromTemplate(`
    You are a Star Wars fanfiction writer. Write an entertaining and comedic story about the Star Wars character ${userInput} who visits a planet or city that's ${place}
    `)
	]);

	const chain = RunnableSequence.from([
		{
			character: vectorStoreRetriever.pipe(formatDocsToChars),
			userInput: new RunnablePassthrough(),
		},
		prompt,
		chat,
		new StringOutputParser(),
	]);

	const chainStream = await chain.stream(userInput);
	return streamText(c, async (stream) => {
		for await (const token of chainStream) {
			stream.write(token);
		}
	});
});

// Quick test to make sure we are getting back documents we want
app.get('/search', async (c) => {
	const query = c.req.query('q');
	const embeddings = new CloudflareWorkersAIEmbeddings({
		binding: c.env.AI,
		modelName: EMBEDDINGS_MODEL,
	});
	const store = new CloudflareVectorizeStore(embeddings, {
		index: c.env.VECTORIZE_INDEX,
	});
	const results = await store.similaritySearch(query, 5);
	console.log(results);
	return Response.json(results);
});

app.get('/loadbucket', async (c) => {
	const embeddings = new CloudflareWorkersAIEmbeddings({
		binding: c.env.AI,
		modelName: EMBEDDINGS_MODEL,
	});
	const store = new CloudflareVectorizeStore(embeddings, {
		index: c.env.VECTORIZE_INDEX,
	});
	console.log('Getting characters from R2');
	const characters = await getCharacters(c.env.starwarscharacters);
	console.log(`Retrieved ${characters.length}`);
	// Chunk and split
	const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 200, chunkOverlap: 10 });
	for (const c of characters) {
		const docs = await splitter.createDocuments([c.name], [{ species: c.species, homeworld: c.homeworld }]);
		console.log(`${c.title} created ${docs.length} docs`);
		console.log('First one:', JSON.stringify(docs[0]));
		console.log(`Adding to Vectorize`);
		// FIXME: Vectorize doesn't like this loc `object` type
		//docs.forEach((doc) => delete doc.metadata.loc);
		const indexedIds = await store.addDocuments(docs);
		console.log(`Inserted ${JSON.stringify(indexedIds)}`);
	}
});

export default app;
