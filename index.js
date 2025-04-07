/*
 * =============================================================================
 *   FULL USABLE CODE - Salesforce Activity Summarizer (Bulk Query Polling)
 * =============================================================================
 *
 * Enhanced Node.js Express application for generating Salesforce activity summaries
 * using OpenAI Assistants. Optimized for memory usage on platforms like Heroku
 * using Salesforce Bulk Query API Job Polling.
 *
 * Features:
 * - Salesforce Bulk Query API job polling to handle large datasets without
 *   loading all records into memory at once. Fetches results in batches.
 * - Per-task OpenAI Assistant creation/deletion for isolation.
 * - Asynchronous processing with immediate acknowledgement (202 Accepted).
 * - Salesforce integration (bulk query fetch, bulk save).
 * - OpenAI Assistants API V2 usage with Function Calling.
 * - Dynamic function schema loading (default or from request).
 * - Dynamic tool_choice to force specific function calls (monthly/quarterly).
 * - Conditional input method for AI: Direct JSON or File Upload.
 * - Incremental processing: Generates & saves monthly summaries from result
 *   batches, then aggregates & saves quarterly summaries.
 * - Robust error handling and callback mechanism.
 * - Temporary file management for AI file uploads.
 * - Data truncation options for large fields.
 * - Sub-batching within monthly processing for finer memory control.
 */

// --- Dependencies ---
const express = require('express');
const jsforce = require('jsforce');
const dotenv = require('dotenv');
const { OpenAI, NotFoundError } = require("openai");
const fs = require("fs-extra");
const path = require("path");
const axios = require("axios");
// Note: stream and stream/promises are built-in, no separate install needed
const { Readable } = require('stream');
const { pipeline } = require('stream/promises');

// --- Load Environment Variables ---
dotenv.config();

// --- Configuration Constants ---
const SF_LOGIN_URL = process.env.SF_LOGIN_URL;
const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";
const TIMELINE_SUMMARY_OBJECT_API_NAME = "Timeline_Summary__c"; // Your Salesforce object API name
const DIRECT_INPUT_THRESHOLD = 1500; // Max activities for direct JSON input to AI
const PROMPT_LENGTH_THRESHOLD = 200000; // Max chars for AI prompt before switching to file upload
const TEMP_FILE_DIR = path.join(__dirname, 'temp_files'); // Directory for temporary files
const BULK_QUERY_BATCH_SIZE = 500; // Process records in sub-batches within a month's results
const DESCRIPTION_TRUNCATE_LENGTH = 1000; // Max chars for Description field passed to AI
const SUBJECT_TRUNCATE_LENGTH = 250;    // Max chars for Subject field passed to AI
const BULK_API_POLL_TIMEOUT = 300000; // 5 minutes timeout for Bulk API job polling
const BULK_API_POLL_INTERVAL = 15000; // 15 seconds interval for Bulk API job polling


// --- Environment Variable Validation ---
if (!SF_LOGIN_URL || !OPENAI_API_KEY) {
    console.error("FATAL ERROR: Missing required environment variables (SF_LOGIN_URL, OPENAI_API_KEY).");
    process.exit(1);
}

// --- Default OpenAI Function Schemas ---
// (Ensure parameter descriptions match truncated fields if applicable)
const defaultFunctions = [
    {
      "name": "generate_monthly_activity_summary",
      "description": "Generates a structured monthly sales activity summary with insights and categorization based on provided activity data (fields like Description/Subject may be truncated). Apply sub-theme segmentation within activityMapping.",
      "parameters": {
        "type": "object",
        "properties": {
          "summary": {
            "type": "string",
            "description": "HTML summary for the month. MUST have one H1 header 'Sales Activity Summary for {Month} {Year}' (no bold) followed by a UL list of key insights."
          },
          "activityMapping": {
            "type": "object",
            "description": "Activities categorized under predefined themes. Each category key holds an array where each element represents a distinct sub-theme identified within that category.",
            "properties": {
              "Key Themes of Customer Interaction": {
                "type": "array",
                "description": "An array where each element represents a distinct sub-theme identified in customer interactions (e.g., 'Pricing', 'Support'). Generate multiple elements if multiple distinct themes are found.",
                "items": {
                  "type": "object",
                  "description": "Represents a single, specific sub-theme identified within 'Key Themes'. Contains a focused summary and ONLY the activities related to this sub-theme.",
                  "properties": {
                    "Summary": {
                      "type": "string",
                      "description": "A concise summary describing this specific sub-theme ONLY (e.g., 'Discussions focused on contract renewal terms')."
                    },
                    "ActivityList": {
                      "type": "array",
                      "description": "A list containing ONLY the activities specifically relevant to this sub-theme.",
                      "items": {
                        "type": "object",
                        "properties": {
                          "Id": { "type": "string", "description": "Salesforce Id of the specific activity" },
                          "LinkText": { "type": "string", "description": "'MMM DD YYYY: Short Description (max 50 chars)' - Generate from ActivityDate, Subject, Description (subject/desc may be truncated)." },
                          "ActivityDate": { "type": "string", "description": "Activity Date in 'YYYY-MM-DD' format. Must belong to the summarized month/year." }
                        },
                        "required": ["Id", "LinkText", "ActivityDate"],
                        "additionalProperties": false
                      }
                    }
                  },
                  "required": ["Summary", "ActivityList"],
                  "additionalProperties": false
                }
              },
              "Tone and Purpose of Interaction": {
                "type": "array",
                "description": "An array where each element represents a distinct tone or strategic intent identified (e.g., 'Information Gathering', 'Negotiation'). Generate multiple elements if distinct patterns are found.",
                 "items": {
                  "type": "object",
                  "description": "Represents a single, specific tone/purpose pattern. Contains a focused summary and ONLY the activities exhibiting this pattern.",
                  "properties": {
                     "Summary": { "type": "string", "description": "A concise summary describing this specific tone/purpose ONLY." },
                     "ActivityList": {
                       "type": "array",
                       "description": "A list containing ONLY the activities specifically exhibiting this tone/purpose.",
                       "items": {
                          "type": "object",
                          "properties": {
                            "Id": { "type": "string", "description": "Salesforce Id of the specific activity" },
                            "LinkText": { "type": "string", "description": "'MMM DD YYYY: Short Description (max 50 chars)' - Generate from ActivityDate, Subject, Description (subject/desc may be truncated)." },
                            "ActivityDate": { "type": "string", "description": "Activity Date in 'YYYY-MM-DD' format. Must belong to the summarized month/year." }
                          },
                          "required": ["Id", "LinkText", "ActivityDate"],
                          "additionalProperties": false
                        }
                      }
                   },
                   "required": ["Summary", "ActivityList"], "additionalProperties": false
                 }
              },
              "Recommended Action and Next Steps": {
                "type": "array",
                "description": "An array where each element represents a distinct type of recommended action or next step identified (e.g., 'Schedule Follow-up Demo', 'Send Proposal'). Generate multiple elements if distinct recommendations are found.",
                 "items": {
                   "type": "object",
                   "description": "Represents a single, specific recommended action type. Contains a focused summary and ONLY the activities leading to this recommendation.",
                   "properties": {
                      "Summary": { "type": "string", "description": "A concise summary describing this specific recommendation type ONLY." },
                      "ActivityList": {
                        "type": "array",
                        "description": "A list containing ONLY the activities specifically related to this recommendation.",
                        "items": {
                          "type": "object",
                          "properties": {
                            "Id": { "type": "string", "description": "Salesforce Id of the specific activity" },
                            "LinkText": { "type": "string", "description": "'MMM DD YYYY: Short Description (max 50 chars)' - Generate from ActivityDate, Subject, Description (subject/desc may be truncated)." },
                            "ActivityDate": { "type": "string", "description": "Activity Date in 'YYYY-MM-DD' format. Must belong to the summarized month/year." }
                          },
                          "required": ["Id", "LinkText", "ActivityDate"],
                          "additionalProperties": false
                        }
                      }
                   },
                   "required": ["Summary", "ActivityList"], "additionalProperties": false
                 }
              }
            },
            "required": ["Key Themes of Customer Interaction", "Tone and Purpose of Interaction", "Recommended Action and Next Steps"]
          },
          "activityCount": {
            "type": "integer",
            "description": "Total number of activities processed for the month/batch (matching the input count)."
          }
        },
        "required": ["summary", "activityMapping", "activityCount"]
      }
    },
    {
      "name": "generate_quarterly_activity_summary",
      "description": "Aggregates provided monthly summaries (as JSON) into a structured quarterly report for a specific quarter, grouped by year.",
      "parameters": {
        "type": "object",
        "properties": {
          "yearlySummary": {
            "type": "array",
            "description": "Quarterly summary data, grouped by year. Should typically contain only one year based on input.",
            "items": {
              "type": "object",
              "properties": {
                "year": {
                  "type": "integer",
                  "description": "The calendar year of the quarter being summarized."
                },
                "quarters": {
                  "type": "array",
                  "description": "List containing the summary for the single quarter being processed.",
                  "items": {
                    "type": "object",
                    "properties": {
                      "quarter": {
                        "type": "string",
                        "description": "Quarter identifier (e.g., Q1, Q2, Q3, Q4) corresponding to the input monthly data."
                      },
                      "summary": {
                        "type": "string",
                        "description": "HTML summary for the quarter. MUST have one H1 header 'Sales Activity Summary for {Quarter} {Year}' (no bold) followed by a UL list of key aggregated insights."
                      },
                      "activityMapping": {
                        "type": "array",
                        "description": "Aggregated activities categorized under predefined themes for the entire quarter.",
                        "items": {
                          "type": "object",
                          "description": "Represents one main category for the quarter, containing an aggregated summary and a consolidated list of all relevant activities.",
                          "properties": {
                            "category": {
                              "type": "string",
                              "description": "Category name (Must be one of 'Key Themes of Customer Interaction', 'Tone and Purpose of Interaction', 'Recommended Action and Next Steps')."
                            },
                            "summary": {
                              "type": "string",
                              "description": "Aggregated summary synthesizing findings for this category across the entire quarter, highlighting key quarterly sub-themes identified."
                            },
                            "activityList": {
                              "type": "array",
                              "description": "Consolidated list of ALL activities for this category from the input monthly summaries for this quarter.",
                              "items": {
                                "type": "object",
                                "properties": {
                                  "id": { "type": "string", "description": "Salesforce Activity ID (copied from monthly input)." },
                                  "linkText": { "type": "string", "description": "'MMM DD YYYY: Short Description' (copied from monthly input)." },
                                  "ActivityDate": { "type": "string", "description": "Activity Date in 'YYYY-MM-DD' format (copied from monthly input)." }
                                },
                                "required": ["id", "linkText", "ActivityDate"],
                                "additionalProperties": false
                              }
                            }
                          },
                          "required": ["category", "summary", "activityList"],
                          "additionalProperties": false
                        }
                      },
                      "activityCount": {
                        "type": "integer",
                        "description": "Total number of unique activities aggregated for the quarter from monthly inputs."
                      },
                      "startdate": {
                        "type": "string",
                        "description": "Start date of the quarter being summarized (YYYY-MM-DD)."
                      }
                    },
                    "required": ["quarter", "summary", "activityMapping", "activityCount", "startdate"]
                  }
                }
              },
              "required": ["year", "quarters"]
            }
          }
        },
        "required": ["yearlySummary"]
      }
    }
];

// --- OpenAI Client Initialization ---
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// --- Express Application Setup ---
const app = express();
// Adjust limits if needed, but 10mb should be sufficient for most requests/responses
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// --- Server Startup ---
app.listen(PORT, async () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Using OpenAI Model: ${OPENAI_MODEL}`);
    console.log(`Direct JSON input threshold: ${DIRECT_INPUT_THRESHOLD} activities`);
    console.log(`Prompt length threshold: ${PROMPT_LENGTH_THRESHOLD} characters`);
    console.log(`Bulk Query Sub-Batch Size: ${BULK_QUERY_BATCH_SIZE}`);
    console.log(`Description Truncation: ${DESCRIPTION_TRUNCATE_LENGTH} chars`);
    console.log(`Subject Truncation: ${SUBJECT_TRUNCATE_LENGTH} chars`);
    console.log(`Bulk API Poll Timeout: ${BULK_API_POLL_TIMEOUT}ms`);
    console.log(`Bulk API Poll Interval: ${BULK_API_POLL_INTERVAL}ms`);
    try {
        // Ensure temp directory exists on startup
        await fs.ensureDir(TEMP_FILE_DIR);
        console.log(`Temporary file directory ensured at: ${TEMP_FILE_DIR}`);
    } catch (err) {
        console.error(`FATAL: Could not create temporary directory ${TEMP_FILE_DIR}. Exiting.`, err);
        process.exit(1);
    }
});

// --- Main API Endpoint ---
app.post('/generatesummary', async (req, res) => {
    const requestLogPrefix = "[Request]"; // Prefix for logs related to this specific request handling
    console.log(`${requestLogPrefix} Received /generatesummary request`);

    // --- Authorization ---
    const authHeader = req.headers["authorization"];
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
        console.warn(`${requestLogPrefix} Unauthorized request: Missing or invalid Bearer token.`);
        return res.status(401).json({ error: "Unauthorized" });
    }
    const accessToken = authHeader.split(" ")[1];

    // --- Request Body Destructuring & Validation ---
    const {
        accountId,
        callbackUrl,
        userPrompt, // Template for monthly prompt
        userPromptQtr, // Template for quarterly prompt
        queryText, // SOQL query to fetch activities
        summaryMap, // Optional JSON string map of existing summary records
        loggedinUserId,
        qtrJSON, // Optional override for quarterly function schema
        monthJSON // Optional override for monthly function schema
    } = req.body;

    if (!accountId || !callbackUrl || !accessToken || !queryText || !userPrompt || !userPromptQtr || !loggedinUserId) {
        console.warn(`${requestLogPrefix} Bad Request: Missing required parameters.`);
        return res.status(400).send({ error: "Missing required parameters (accountId, callbackUrl, accessToken, queryText, userPrompt, userPromptQtr, loggedinUserId)" });
    }

     // Validate SOQL basic structure and REQUIRED ordering
     if (!queryText.toLowerCase().includes('select') || !queryText.toLowerCase().includes('from') || !queryText.toLowerCase().includes('activitydate')) {
         console.warn(`${requestLogPrefix} Bad Request: queryText seems invalid or missing ActivityDate field.`);
         return res.status(400).send({ error: "queryText must be a valid SOQL query including the ActivityDate field." });
     }
     // Crucial for bulk query result processing logic: Ensure ORDER BY ActivityDate ASC is present
     if (!queryText.toLowerCase().includes('order by activitydate asc')) {
         console.warn(`${requestLogPrefix} Bad Request: queryText MUST include 'ORDER BY ActivityDate ASC' for processing logic to work correctly.`);
         return res.status(400).send({ error: "queryText must include 'ORDER BY ActivityDate ASC'." });
     }


    // --- Parse Optional JSON Inputs Safely ---
    let summaryRecordsMap = {};
    let monthlyFuncSchema = defaultFunctions.find(f => f.name === 'generate_monthly_activity_summary');
    let quarterlyFuncSchema = defaultFunctions.find(f => f.name === 'generate_quarterly_activity_summary');

    try {
        if (summaryMap) {
            summaryRecordsMap = Object.entries(JSON.parse(summaryMap)).map(([key, value]) => ({ key, value }));
        }
        if (monthJSON) {
            monthlyFuncSchema = JSON.parse(monthJSON);
            if (!monthlyFuncSchema || monthlyFuncSchema.name !== 'generate_monthly_activity_summary') {
                throw new Error("Provided monthJSON schema is invalid or missing the correct name property.");
            }
            console.log(`${requestLogPrefix} Using custom monthly function schema from request.`);
        }
         if (qtrJSON) {
            quarterlyFuncSchema = JSON.parse(qtrJSON);
            if (!quarterlyFuncSchema || quarterlyFuncSchema.name !== 'generate_quarterly_activity_summary') {
                 throw new Error("Provided qtrJSON schema is invalid or missing the correct name property.");
            }
            console.log(`${requestLogPrefix} Using custom quarterly function schema from request.`);
        }
    } catch (e) {
        console.error(`${requestLogPrefix} Failed to parse JSON input from request body:`, e);
        return res.status(400).send({ error: `Invalid JSON provided in summaryMap, monthJSON, or qtrJSON. ${e.message}` });
    }

    // --- Ensure Schemas are Available ---
    if (!monthlyFuncSchema || !quarterlyFuncSchema) {
        console.error(`${requestLogPrefix} FATAL: Default function schemas could not be loaded or found.`);
        return res.status(500).send({ error: "Internal server error: Could not load function schemas."});
    }

    // --- Acknowledge Request (202 Accepted) ---
    // Send response immediately, process runs in background
    res.status(202).json({ status: 'processing', message: 'Summary generation initiated using Bulk API. You will receive a callback.' });
    console.log(`${requestLogPrefix} Initiating summary processing for Account ID: ${accountId}`);

    // --- Start Asynchronous Processing ---
    // Run processSummary without awaiting it here. Handle errors within the function via callback.
    processSummary( // Call the Bulk Query Job Polling version
        accountId,
        accessToken,
        callbackUrl,
        userPrompt,
        userPromptQtr,
        queryText, // Pass the validated SOQL query
        summaryRecordsMap,
        loggedinUserId,
        monthlyFuncSchema,
        quarterlyFuncSchema
    ).catch(async (error) => {
        // This catch block handles errors thrown BEFORE the main try/catch inside processSummary
        console.error(`${requestLogPrefix} Unhandled error during background processing setup for ${accountId}:`, error.stack);
        try {
            // Attempt to send a failure callback if setup fails badly
            await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Unhandled processing setup error: ${error.message}`);
        } catch (callbackError) {
            console.error(`${requestLogPrefix} Failed to send error callback after unhandled setup exception for ${accountId}:`, callbackError);
        }
    });
});


// --- Helper Function to Get Quarter from Month Index ---
function getQuarterFromMonthIndex(monthIndex) {
    if (monthIndex >= 0 && monthIndex <= 2) return 'Q1';
    if (monthIndex >= 3 && monthIndex <= 5) return 'Q2';
    if (monthIndex >= 6 && monthIndex <= 8) return 'Q3';
    if (monthIndex >= 9 && monthIndex <= 11) return 'Q4';
    return 'Unknown'; // Should not happen with valid index
}


// --- Asynchronous Summary Processing Logic (Refactored for Bulk Query Job Polling) ---
async function processSummary(
    accountId,
    accessToken,
    callbackUrl,
    userPromptMonthlyTemplate,
    userPromptQuarterlyTemplate,
    queryText, // SOQL query (MUST include ORDER BY ActivityDate ASC)
    summaryRecordsMap,
    loggedinUserId,
    monthlyFuncSchema,
    quarterlyFuncSchema
) {
    const logPrefix = `[Process ${accountId}]`;
    console.log(`${logPrefix} Starting processSummary (Bulk Query Job Polling)...`);
    const conn = new jsforce.Connection({
        instanceUrl: SF_LOGIN_URL,
        accessToken: accessToken,
        maxRequest: 10, // Retry requests on transient errors
        // Use configured poll settings
        pollTimeout: BULK_API_POLL_TIMEOUT,
        pollInterval: BULK_API_POLL_INTERVAL,
    });

    // Stores AI output from monthly summaries for quarterly aggregation
    const quarterlyInputs = {};
    const monthMap = { january: 0, february: 1, march: 2, april: 3, may: 4, june: 5, july: 6, august: 7, september: 8, october: 9, november: 10, december: 11 };
    let overallStatus = "Success"; // Assume success until an error occurs
    let failureMessages = [];
    let recordCount = 0; // Count records processed from results
    let currentYear = null;
    let currentMonth = null;
    let currentMonthActivities = []; // Holds activities for the current month/sub-batch being processed
    let bulkJob = null; // Reference to the bulk job for potential cleanup

    // --- Helper Function to process a completed month's batch ---
    // This function takes a batch of activities for a single month and processes it
    async function processAndSaveMonthBatch(year, month, activities) {
        const activityCount = activities.length;
        if (activityCount === 0) return; // Safety check

        const batchLogPrefix = `${logPrefix} -> [Batch ${month} ${year}]`;
        console.log(`${batchLogPrefix} Processing ${activityCount} activities`);
        const monthIndex = monthMap[month.toLowerCase()];
        if (monthIndex === undefined) {
            console.warn(`${batchLogPrefix} Could not map month name: ${month}. Skipping batch.`);
            return;
        }
        const startDate = new Date(Date.UTC(year, monthIndex, 1));
        const startDateStr = startDate.toISOString().split('T')[0];
        const userPromptMonthly = userPromptMonthlyTemplate.replace('{{YearMonth}}', `${month} ${year}`);

        let monthlyAiOutput = null;
        let monthlyAssistant = null;

        try {
            // Create a unique name for the assistant for better tracking if needed
            const assistantName = `Monthly Summarizer ${accountId} ${year}-${month}-${Date.now()}`;
            console.log(`${batchLogPrefix}   Creating Assistant: ${assistantName}`);
            monthlyAssistant = await openai.beta.assistants.create({
               name: assistantName,
               instructions: "You are an AI assistant specialized in analyzing raw Salesforce activity data for a single month (fields like Description/Subject may be truncated) and generating structured JSON summaries using the provided function 'generate_monthly_activity_summary'. Apply sub-theme segmentation. Focus on key themes, tone, and actions.",
               tools: [{ type: "file_search" }, { type: "function", "function": monthlyFuncSchema }],
               model: OPENAI_MODEL, // Use configured model
            });
            console.log(`${batchLogPrefix}   Created Assistant ID: ${monthlyAssistant.id}`);

            // Generate the summary using the AI
            monthlyAiOutput = await generateSummary(activities, openai, monthlyAssistant.id, userPromptMonthly, monthlyFuncSchema, accountId);
            console.log(`${batchLogPrefix}   Generated monthly summary AI output.`);

            // Prepare the data structure for saving to Salesforce
            const monthlyForSalesforce = {
               [year]: {
                   [month]: {
                       summary: JSON.stringify(monthlyAiOutput), // Store the full AI JSON response
                       summaryDetails: monthlyAiOutput?.summary || '', // Extract the HTML summary part
                       count: activityCount, // Number of activities in this specific batch
                       startdate: startDateStr
                   }
               }
            };

            // Save the generated summary to Salesforce
            console.log(`${batchLogPrefix}   Saving monthly summary to Salesforce...`);
            await createTimileSummarySalesforceRecords(conn, monthlyForSalesforce, accountId, 'Monthly', summaryRecordsMap);
            console.log(`${batchLogPrefix}   Saved monthly summary.`);

            // Store the AI output needed for the quarterly aggregation step
            const quarter = getQuarterFromMonthIndex(monthIndex);
            const quarterKey = `${year}-${quarter}`;
            if (!quarterlyInputs[quarterKey]) {
                quarterlyInputs[quarterKey] = { monthSummaries: [], year: parseInt(year), quarter: quarter };
            }
            quarterlyInputs[quarterKey].monthSummaries.push(monthlyAiOutput);

        } catch (monthError) {
            // Log errors encountered during this batch processing
            console.error(`${batchLogPrefix} Error processing batch:`, monthError.stack);
            overallStatus = "Failed"; // Mark the entire process as failed if any batch errors out
            failureMessages.push(`Failed processing ${month} ${year}: ${monthError.message}`);
        } finally {
            // Ensure the temporary OpenAI assistant is deleted
            if (monthlyAssistant?.id) {
                try {
                    console.log(`${batchLogPrefix}   Deleting monthly assistant ${monthlyAssistant.id}`);
                    await openai.beta.assistants.del(monthlyAssistant.id);
                    console.log(`${batchLogPrefix}   Deleted monthly assistant ${monthlyAssistant.id}`);
                } catch (delError) {
                    console.warn(`${batchLogPrefix}   Could not delete monthly assistant ${monthlyAssistant.id}:`, delError.message || delError);
                 }
            }
            // Help garbage collector by nullifying references
            activities = null;
            monthlyAiOutput = null;
        }
    }
    // --- End of processAndSaveMonthBatch ---

    // --- Main Processing Logic ---
    try {
        // 1. Execute Bulk Query Job and Process Results via Polling
        console.log(`${logPrefix} Starting Bulk API Query job... Query: ${queryText.substring(0,150)}...`);

        // Create the Bulk API job for the 'query' operation.
        // Use a relevant object name; it's mainly for context, the query defines the actual object.
        // Using 'Account' as a placeholder, adjust if needed, though it matters less for 'query'.
        bulkJob = conn.bulk.createJob("Account", "query"); // Changed object to Account for example

        // Create a batch within the job containing the SOQL query
        const batch = bulkJob.createBatch();

        // Listen for the 'queue' event to know the batch is submitted
        // Use a Promise to wait for the entire batch processing (polling + result fetching)
        await new Promise((resolve, reject) => {
            batch.on("queue", (batchInfo) => {
                console.log(`${logPrefix} Bulk query batch queued. Batch ID: ${batchInfo.id}, Job ID: ${batchInfo.jobId}`);

                // Start polling for the batch completion status using the configured interval and timeout
                console.log(`${logPrefix} Starting to poll batch status (Interval: ${conn.pollInterval}ms, Timeout: ${conn.pollTimeout}ms)...`);
                batch.poll(conn.pollInterval, conn.pollTimeout); // jsforce handles the polling loop

                // Listen for the 'response' event, which fires when polling detects completion
                batch.on("response", async (results) => {
                    console.log(`${logPrefix} Bulk query job finished. Received ${results.length} result locators (batches). Processing results...`);
                    recordCount = 0; // Reset count for records fetched from results

                    try {
                        // Iterate through each result locator returned by the completed job
                        for (const resultInfo of results) {
                            const resultBatchLogPrefix = `${logPrefix} -> [Result Batch ${resultInfo.id}]`;
                            console.log(`${resultBatchLogPrefix} Fetching results...`);
                            // Retrieve the actual records associated with this result batch ID
                            // This fetches one chunk of the total query result from SF storage
                            const records = await conn.bulk.job(resultInfo.jobId).batch(resultInfo.id).result();
                            console.log(`${resultBatchLogPrefix} Fetched ${records.length} records.`);

                            // --- Process the fetched records from this result batch ---
                            for (const record of records) {
                                recordCount++;
                                // Basic validation
                                if (!record.ActivityDate) {
                                    console.warn(`[Result Record ${recordCount} ${accountId}] Skipping activity (ID: ${record.Id || 'Unknown'}) - missing ActivityDate.`);
                                    continue;
                                }
                                try {
                                    const date = new Date(record.ActivityDate);
                                    if (isNaN(date.getTime())) {
                                         console.warn(`[Result Record ${recordCount} ${accountId}] Skipping activity (ID: ${record.Id || 'Unknown'}) - invalid ActivityDate: ${record.ActivityDate}`);
                                         continue;
                                     }

                                    const year = date.getUTCFullYear();
                                    const month = date.toLocaleString('en-US', { month: 'long', timeZone: 'UTC' });

                                    // --- Month Change Detection (Requires ORDER BY ActivityDate ASC) ---
                                    // Check if the current record belongs to a new month/year compared to the previous one
                                    if (year !== currentYear || month !== currentMonth) {
                                        // If we have accumulated activities from the *previous* month, process them now.
                                        if (currentMonthActivities.length > 0) {
                                            await processAndSaveMonthBatch(currentYear, currentMonth, currentMonthActivities);
                                            currentMonthActivities = []; // Clear the batch after processing
                                        }
                                        // Update the state to the new month/year
                                        currentYear = year;
                                        currentMonth = month;
                                    }

                                    // Add essential, TRUNCATED data to the current month's activity batch
                                    currentMonthActivities.push({
                                        Id: record.Id,
                                        Description: record.Description?.substring(0, DESCRIPTION_TRUNCATE_LENGTH) || null,
                                        Subject: record.Subject?.substring(0, SUBJECT_TRUNCATE_LENGTH) || null,
                                        ActivityDate: record.ActivityDate // Store the original date string
                                        // Add other essential fields if needed by AI
                                    });

                                    // --- Sub-Batching within a Month's Results ---
                                    // If the accumulated activities for the current month reach the sub-batch size,
                                    // process them immediately to keep memory usage low even within large result batches.
                                    if (currentMonthActivities.length >= BULK_QUERY_BATCH_SIZE) {
                                         console.log(`[Result ${accountId}] Processing sub-batch for ${currentMonth} ${currentYear} (size ${currentMonthActivities.length})`);
                                         await processAndSaveMonthBatch(currentYear, currentMonth, currentMonthActivities);
                                         currentMonthActivities = []; // Clear the processed sub-batch
                                    }

                                } catch (recordError) {
                                    // Log errors processing individual records but continue with the next record
                                    console.error(`[Result Record ${recordCount} ${accountId}] Error processing record (ID: ${record.Id || 'Unknown'}):`, recordError.stack);
                                    // Optionally mark status as partial success here
                                }
                            } // --- End of record processing loop for this result batch ---
                        } // --- End of loop iterating through result locators ---

                        // Process the very last accumulated batch after all result sets have been handled
                        if (currentMonthActivities.length > 0) {
                            console.log(`${logPrefix} Processing final accumulated batch for ${currentMonth} ${currentYear} (${currentMonthActivities.length} records)`);
                            await processAndSaveMonthBatch(currentYear, currentMonth, currentMonthActivities);
                            currentMonthActivities = []; // Clear the final batch
                        }
                        console.log(`${logPrefix} Finished processing all bulk query results. Total records processed: ${recordCount}.`);
                        resolve(); // Resolve the main promise indicating successful processing of results

                    } catch (fetchError) {
                         // Catch errors during the result fetching loop
                         console.error(`${logPrefix} Error fetching or processing bulk results:`, fetchError.stack);
                         failureMessages.push(`Failed to fetch/process results: ${fetchError.message}`);
                         reject(fetchError); // Reject the main promise
                    }
                }); // End of "response" handler

                // Listen for errors during the polling process itself
                batch.on("error", (err) => {
                    console.error(`${logPrefix} Error during bulk query batch polling/processing:`, err);
                    failureMessages.push(`Bulk query batch error: ${err.message}`);
                    reject(err); // Reject the main promise on batch error
                });

            }); // End of "queue" handler

            // Handle errors that might occur when initially trying to queue the batch
            batch.on("error", (err) => {
                 console.error(`${logPrefix} Error queuing bulk query batch:`, err);
                 failureMessages.push(`Failed to queue bulk batch: ${err.message}`);
                 reject(err); // Reject the main promise
            });

            // Execute the batch (send the SOQL query to Salesforce)
            console.log(`${logPrefix} Executing batch...`);
            batch.execute(queryText);

        }); // --- End of Bulk Processing Promise ---

        // --- Close the Bulk Job (Best Practice) ---
        // No need to await closure if we don't need to guarantee it before proceeding
        if (bulkJob && bulkJob.id) {
            console.log(`${logPrefix} Attempting to close bulk job ID: ${bulkJob.id}`);
            bulkJob.close().then(() => {
                console.log(`${logPrefix} Bulk job ${bulkJob.id} closed successfully.`);
            }).catch(closeErr => {
                // Log warning but don't fail the overall process just for closing error
                console.warn(`${logPrefix} Failed to close bulk job ID ${bulkJob.id}:`, closeErr.message);
            });
        }


        // 2. Process Quarters Incrementally (Based on aggregated `quarterlyInputs`)
        // This part remains largely the same as the previous version, as it operates on the
        // already processed monthly summaries stored in memory (which should be manageable).
        console.log(`${logPrefix} Processing ${Object.keys(quarterlyInputs).length} quarters based on processed data...`);
        for (const quarterKey in quarterlyInputs) {
             const quarterlyLogPrefix = `${logPrefix} -> [Quarter ${quarterKey}]`;
             let { monthSummaries, year, quarter } = quarterlyInputs[quarterKey];
             const numMonthlySummaries = monthSummaries?.length || 0;
             console.log(`${quarterlyLogPrefix} Generating summary using ${numMonthlySummaries} monthly summaries...`);

             if (numMonthlySummaries === 0) {
                 console.warn(`${quarterlyLogPrefix} Skipping as it has no associated monthly summaries.`);
                 continue;
             }

             // Prepare prompt with aggregated monthly JSON data
             const quarterlyInputDataString = JSON.stringify(monthSummaries, null, 2);
             const userPromptQuarterly = `${userPromptQuarterlyTemplate.replace('{{Quarter}}', quarter).replace('{{Year}}', year)}\n\nAggregate the following monthly summary data provided below for ${quarterKey}:\n\`\`\`json\n${quarterlyInputDataString}\n\`\`\``;

             // Check if the combined monthly data is excessively large
             if(quarterlyInputDataString.length > PROMPT_LENGTH_THRESHOLD * 2) { // Arbitrary check, adjust threshold if needed
                console.warn(`${quarterlyLogPrefix} Combined monthly summaries JSON (${quarterlyInputDataString.length} chars) is very large. Consider optimizing monthly AI output structure or using file input for quarterly stage if memory issues arise here.`);
             }

             let quarterlyAiOutput = null;
             let quarterlyAssistant = null;

             try {
                 // Create temporary assistant for this quarterly summary
                 const assistantName = `Quarterly Summarizer ${accountId} ${quarterKey}-${Date.now()}`;
                 console.log(`${quarterlyLogPrefix}   Creating Assistant: ${assistantName}`);
                 quarterlyAssistant = await openai.beta.assistants.create({
                      name: assistantName,
                      instructions: "You are an AI assistant specialized in aggregating pre-summarized monthly Salesforce activity data (provided as JSON in the prompt) into a structured quarterly JSON summary for a specific quarter using the provided function 'generate_quarterly_activity_summary'. Consolidate insights and activity lists accurately.",
                      tools: [{ type: "file_search" }, { type: "function", "function": quarterlyFuncSchema }],
                      model: OPENAI_MODEL,
                 });
                 console.log(`${quarterlyLogPrefix}   Created Assistant ID: ${quarterlyAssistant.id}`);

                 // Generate the quarterly summary (passing null activities, data is in prompt)
                 quarterlyAiOutput = await generateSummary(null, openai, quarterlyAssistant.id, userPromptQuarterly, quarterlyFuncSchema, accountId);
                 console.log(`${quarterlyLogPrefix}   Generated quarterly summary AI output.`);

                 // Transform the AI output into Salesforce format
                 const transformedQuarter = transformQuarterlyStructure(quarterlyAiOutput);

                 // Validate transformation and save to Salesforce
                 if (transformedQuarter && transformedQuarter[year] && transformedQuarter[year][quarter]) {
                      const quarterlyForSalesforce = { [year]: { [quarter]: transformedQuarter[year][quarter] } };
                      console.log(`${quarterlyLogPrefix}   Saving quarterly summary to Salesforce...`);
                      await createTimileSummarySalesforceRecords(conn, quarterlyForSalesforce, accountId, 'Quarterly', summaryRecordsMap);
                      console.log(`${quarterlyLogPrefix}   Saved quarterly summary.`);
                 } else {
                      // Log warning if transformation failed
                      console.warn(`${quarterlyLogPrefix} Quarterly summary generated by AI but transform/validation failed. Skipping save.`);
                      if (overallStatus !== "Failed") { overallStatus = "Partial Success"; } // Mark as partial success
                      failureMessages.push(`Failed to transform/validate AI output for ${quarterKey}.`);
                 }
             } catch (quarterlyError) {
                  // Log errors during quarterly processing
                  console.error(`${quarterlyLogPrefix} Failed to generate or save quarterly summary:`, quarterlyError.stack);
                  overallStatus = "Failed"; // Mark overall process as failed
                  failureMessages.push(`Failed processing ${quarterKey}: ${quarterlyError.message}`);
             } finally {
                  // Cleanup Quarterly Assistant
                  if (quarterlyAssistant?.id) {
                      try {
                          console.log(`${quarterlyLogPrefix}   Deleting quarterly assistant ${quarterlyAssistant.id}`);
                          await openai.beta.assistants.del(quarterlyAssistant.id);
                          console.log(`${quarterlyLogPrefix}   Deleted quarterly assistant ${quarterlyAssistant.id}`);
                      } catch (delError) {
                          console.warn(`${quarterlyLogPrefix}   Could not delete quarterly assistant ${quarterlyAssistant.id}:`, delError.message || delError);
                       }
                  }
                  // Release memory references
                  if (quarterlyInputs[quarterKey]) quarterlyInputs[quarterKey].monthSummaries = null;
                  quarterlyAiOutput = null;
                  monthSummaries = null;
             }
        } // --- End of Quarterly Processing Loop ---


        // 3. Send Final Callback Notification
        console.log(`${logPrefix} Process completed.`);
        let finalMessage = overallStatus === "Success" ? "Summary Processed Successfully" : `Processing finished with issues: ${failureMessages.join('; ')}`;
        // Truncate potentially long failure message lists before sending
        finalMessage = finalMessage.length > 1000 ? finalMessage.substring(0, 997) + "..." : finalMessage;
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, overallStatus, finalMessage);

    } catch (error) {
        // Catch errors from the main processing flow (e.g., bulk job setup, promise rejection)
        console.error(`${logPrefix} Critical error during summary processing:`, error.stack);
        // Ensure a failure callback is sent for critical errors
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Critical processing error: ${error.message}`);
        // Attempt to abort/close the job if it exists and a critical error occurred
        if (bulkJob && bulkJob.id) {
            try {
                console.warn(`${logPrefix} Attempting to abort/close bulk job ${bulkJob.id} due to critical error...`);
                // Use abort() if job might be running, close() if likely completed/failed state
                await bulkJob.abort(); // More likely to stop an ongoing job
                console.log(`${logPrefix} Bulk job aborted.`);
            } catch (abortErr) {
                 try { // Fallback to close if abort fails (e.g., job already finished)
                     await bulkJob.close();
                     console.log(`${logPrefix} Bulk job closed.`);
                 } catch(closeErr) {
                    console.warn(`${logPrefix} Failed to abort/close bulk job ${bulkJob.id} after error:`, closeErr.message);
                 }
            }
        }
    } finally {
         console.log(`${logPrefix} processSummary finished execution.`);
         // Optional final GC & memory log
        //  if (global.gc) { console.log(`${logPrefix} Triggering final GC.`); global.gc(); }
        //  console.log(`${logPrefix} Final memory usage:`, process.memoryUsage());
    }
}


// --- OpenAI Summary Generation Function (Includes conditional input) ---
// Handles interaction with OpenAI Assistants API for a given batch/prompt
async function generateSummary(
    activities, // Array of raw activities OR null
    openaiClient,
    assistantId,
    userPrompt,
    functionSchema,
    accountId = 'N/A' // For logging context
) {
    let fileId = null;
    let thread = null;
    let tempFilePath = null;
    let inputMethod = "prompt";
    let logPrefix = `[AI ${accountId} Thread New]`; // Initial log prefix

    try {
        // 1. Create a new Thread for this interaction
        thread = await openaiClient.beta.threads.create();
        logPrefix = `[AI ${accountId} Thread ${thread.id}]`; // Update log prefix with Thread ID
        console.log(`${logPrefix} Created for Assistant ${assistantId}`);

        let finalUserPrompt = userPrompt;
        let messageAttachments = []; // For potential file uploads

        // 2. Determine Input Method (Direct JSON vs. File Upload)
        if (activities && Array.isArray(activities) && activities.length > 0) {
            let potentialFullPrompt;
            let activitiesJsonString;

            try {
                 // Stringify minimally for length check first
                 activitiesJsonString = JSON.stringify(activities);
                 potentialFullPrompt = `${userPrompt}\n\nActivity data:\n\`\`\`json\n${activitiesJsonString}\n\`\`\``;
                 console.log(`${logPrefix} Potential prompt length with direct JSON: ${potentialFullPrompt.length} chars.`);
            } catch(stringifyError) {
                console.error(`${logPrefix} Error stringifying activities for length check:`, stringifyError);
                throw new Error("Failed to stringify activity data for processing.");
            }

            // Use direct JSON if below thresholds
            if (potentialFullPrompt.length < PROMPT_LENGTH_THRESHOLD && activities.length < DIRECT_INPUT_THRESHOLD) {
                inputMethod = "direct JSON";
                // Use pretty-printed JSON in the final prompt
                finalUserPrompt = `${userPrompt}\n\nActivity data:\n\`\`\`json\n${JSON.stringify(activities, null, 2)}\n\`\`\``;
                console.log(`${logPrefix} Using direct JSON input (${activities.length} activities).`);
            } else {
                // Use file upload if thresholds exceeded
                inputMethod = "file upload";
                console.log(`${logPrefix} Using file upload (Activities: ${activities.length} >= ${DIRECT_INPUT_THRESHOLD} or Prompt Length: ${potentialFullPrompt.length} >= ${PROMPT_LENGTH_THRESHOLD}).`);
                finalUserPrompt = userPrompt; // Use base prompt only

                // Convert activities to plain text for the file
                let activitiesText = activities.map((activity, index) => {
                    let desc = activity.Description || 'No Description';
                    let subj = activity.Subject || 'No Subject';
                    return `Activity ${index + 1} (ID: ${activity.Id || 'N/A'}):\n  ActivityDate: ${activity.ActivityDate || 'N/A'}\n  Subject: ${subj}\n  Description: ${desc}`;
                }).join('\n\n---\n\n');

                // Create and write temporary file
                const timestamp = new Date().toISOString().replace(/[:.-]/g, "_");
                const filename = `activities_${accountId}_${timestamp}_${thread.id}.txt`;
                tempFilePath = path.join(TEMP_FILE_DIR, filename);
                await fs.writeFile(tempFilePath, activitiesText);
                console.log(`${logPrefix} Temporary text file generated: ${tempFilePath}`);

                // Upload file to OpenAI
                const fileStream = fs.createReadStream(tempFilePath);
                const uploadResponse = await openaiClient.files.create({ file: fileStream, purpose: "assistants"});
                fileId = uploadResponse.id;
                console.log(`${logPrefix} File uploaded to OpenAI: ${fileId}`);

                // Prepare attachment for the message
                messageAttachments.push({ file_id: fileId, tools: [{ type: "file_search" }] });
                console.log(`${logPrefix} Attaching file ${fileId} with file_search tool.`);
                finalUserPrompt += "\n\nPlease analyze the activity data provided in the attached file."; // Instruct AI
            }
        } else {
             // No activities provided (e.g., quarterly summary)
             console.log(`${logPrefix} No activities array provided. Using prompt content as is.`);
        }

        // 3. Add Message to Thread
        const messagePayload = { role: "user", content: finalUserPrompt };
        if (messageAttachments.length > 0) {
            messagePayload.attachments = messageAttachments;
        }
        const message = await openaiClient.beta.threads.messages.create(thread.id, messagePayload);
        console.log(`${logPrefix} Message added (using ${inputMethod}). ID: ${message.id}`);

        // 4. Run Assistant and Poll for Result
        console.log(`${logPrefix} Starting run, forcing function: ${functionSchema.name}`);
        const run = await openaiClient.beta.threads.runs.createAndPoll(thread.id, {
            assistant_id: assistantId,
            tools: [{ type: "function", function: functionSchema }], // Provide schema for this run
            tool_choice: { type: "function", function: { name: functionSchema.name } }, // Force function call
        });
        console.log(`${logPrefix} Run status: ${run.status}`);

        // 5. Process Run Outcome
        if (run.status === 'requires_action') {
            // Extract and validate tool call arguments
            const toolCalls = run.required_action?.submit_tool_outputs?.tool_calls;
            if (!toolCalls || toolCalls.length === 0 || !toolCalls[0]?.function?.arguments) {
                 console.error(`${logPrefix} Run requires action, but tool call data is missing or invalid.`, run.required_action);
                 throw new Error("Function call was expected but not provided correctly by the Assistant.");
             }
             const toolCall = toolCalls[0];
             if (toolCall.function.name !== functionSchema.name) {
                  console.error(`${logPrefix} Assistant called the wrong function. Expected: ${functionSchema.name}, Got: ${toolCall.function.name}`);
                  throw new Error(`Assistant called the wrong function: ${toolCall.function.name}`);
             }
             const rawArgs = toolCall.function.arguments;
             console.log(`${logPrefix} Function call arguments received for ${toolCall.function.name}. Length: ${rawArgs.length}`);
             try {
                 // Parse the JSON arguments
                 const summaryObj = JSON.parse(rawArgs);
                 console.log(`${logPrefix} Successfully parsed function arguments.`);
                 return summaryObj; // Return the structured data
             } catch (parseError) {
                 console.error(`${logPrefix} Failed to parse function call arguments JSON:`, parseError);
                 console.error(`${logPrefix} Raw arguments received (first 500 chars):`, rawArgs.substring(0, 500));
                 throw new Error(`Failed to parse function call arguments from AI: ${parseError.message}`);
             }
         } else if (run.status === 'completed') {
              // Handle unexpected completion (should have required action)
              console.warn(`${logPrefix} Run completed without requiring function call action, despite tool_choice forcing ${functionSchema.name}.`);
              try { // Log last assistant message for debugging
                const messages = await openaiClient.beta.threads.messages.list(run.thread_id, { order: 'desc', limit: 1 });
                const lastMessageContent = messages.data[0]?.content[0]?.text?.value || "No text content found.";
                console.warn(`${logPrefix} Last message content from Assistant: ${lastMessageContent.substring(0, 500)}...`);
              } catch (msgError) {
                  console.warn(`${logPrefix} Could not retrieve last message for completed run: ${msgError.message}`);
              }
              throw new Error(`Assistant run completed without making the required function call to ${functionSchema.name}.`);
         } else {
             // Handle failed, cancelled, expired runs
             console.error(`${logPrefix} Run failed or ended unexpectedly. Status: ${run.status}`, run.last_error || run.incomplete_details);
             const errorMessage = run.last_error ? `${run.last_error.code}: ${run.last_error.message}` : (run.incomplete_details?.reason || 'Unknown error');
             throw new Error(`Assistant run failed. Status: ${run.status}. Error: ${errorMessage}`);
         }

    } catch (error) {
        // Catch errors from any step in the AI interaction
        logPrefix = `[AI ${accountId} Thread ${thread?.id || 'N/A'}]`; // Ensure prefix is set even if thread creation failed
        console.error(`${logPrefix} Error in generateSummary: ${error.message}`);
        console.error(error.stack); // Log stack trace
        throw error; // Re-throw
    } finally {
        // 6. Cleanup Resources (Temp file, OpenAI file)
        logPrefix = `[AI ${accountId} Thread ${thread?.id || 'Cleanup'}]`; // Use thread ID if available
        if (tempFilePath) {
            try {
                await fs.unlink(tempFilePath);
                console.log(`${logPrefix} Deleted temporary file: ${tempFilePath}`);
            } catch (unlinkError) {
                console.error(`${logPrefix} Error deleting temporary file ${tempFilePath}:`, unlinkError);
            }
        }
        if (fileId) {
            try {
                await openaiClient.files.del(fileId);
                console.log(`${logPrefix} Deleted OpenAI file: ${fileId}`);
            } catch (deleteError) {
                 if (!(deleteError instanceof NotFoundError || deleteError?.status === 404)) {
                    console.error(`${logPrefix} Error deleting OpenAI file ${fileId}:`, deleteError.message || deleteError);
                 } else {
                     console.log(`${logPrefix} OpenAI file ${fileId} already deleted or not found.`);
                 }
            }
        }
        // Optional: Delete thread
        // if (thread?.id) { /* ... delete thread ... */ }
    }
}


// --- Salesforce Record Creation/Update Function (Bulk API) ---
// Saves generated summaries (monthly or quarterly) to Salesforce
async function createTimileSummarySalesforceRecords(conn, summaries, parentId, summaryCategory, summaryRecordsMap) {
    const logPrefix = `[SF Save ${parentId} ${summaryCategory}]`;
    console.log(`${logPrefix} Preparing to save summaries...`);
    let recordsToCreate = [];
    let recordsToUpdate = [];

    // Iterate through the summaries data structure: { year: { periodKey: { summaryData } } }
    for (const year in summaries) {
        for (const periodKey in summaries[year]) {
            const summaryData = summaries[year][periodKey];
            let summaryJsonString = summaryData.summaryJson || summaryData.summary; // Full AI JSON
            let summaryDetailsHtml = summaryData.summaryDetails || ''; // HTML summary
            let startDate = summaryData.startdate; // YYYY-MM-DD
            let count = summaryData.count; // Activity count for this period/batch

             // Attempt to extract HTML summary from JSON if needed
             if (!summaryDetailsHtml && summaryJsonString) {
                try {
                    const parsedJson = JSON.parse(summaryJsonString);
                    summaryDetailsHtml = parsedJson.summary || parsedJson?.yearlySummary?.[0]?.quarters?.[0]?.summary || '';
                } catch (e) {
                    console.warn(`${logPrefix} Could not parse 'summaryJsonString' for ${periodKey} ${year} to extract HTML details.`);
                 }
            }

            // Determine specific fields based on category (Monthly/Quarterly)
            let fyQuarterValue = (summaryCategory === 'Quarterly') ? periodKey : '';
            let monthValue = (summaryCategory === 'Monthly') ? periodKey : '';
            let shortMonth = monthValue ? monthValue.substring(0, 3) : '';
            let summaryMapKey = (summaryCategory === 'Quarterly') ? `${fyQuarterValue} ${year}` : `${shortMonth} ${year}`;
            let existingRecordId = getValueByKey(summaryRecordsMap, summaryMapKey);

            // Prepare the payload for the Salesforce record
            // Ensure API names match your Timeline_Summary__c object fields
            const recordPayload = {
                Account__c: parentId, // Assumes standard Account lookup
                Month__c: monthValue || null,
                Year__c: String(year),
                Summary_Category__c: summaryCategory,
                // Use substring to avoid exceeding Salesforce field length limits (e.g., 131072)
                Summary__c: summaryJsonString ? summaryJsonString.substring(0, 131070) : null,
                Summary_Details__c: summaryDetailsHtml ? summaryDetailsHtml.substring(0, 131070) : null,
                FY_Quarter__c: fyQuarterValue || null,
                Month_Date__c: startDate,
                Number_of_Records__c: count,
            };

             // Basic validation
             if (!recordPayload.Account__c || !recordPayload.Summary_Category__c || !recordPayload.Year__c) {
                 console.warn(`${logPrefix} Skipping record for ${summaryMapKey} - missing Account ID, Category, or Year.`);
                 continue;
             }

            // Decide whether to create or update
            if (existingRecordId) {
                console.log(`${logPrefix}   Queueing update for ${summaryMapKey} (ID: ${existingRecordId})`);
                recordsToUpdate.push({ Id: existingRecordId, ...recordPayload });
            } else {
                console.log(`${logPrefix}   Queueing create for ${summaryMapKey}`);
                recordsToCreate.push(recordPayload);
            }
        }
    }

    // Perform Salesforce DML using Bulk API
    try {
        const options = { allOrNone: false }; // Allow partial success

        // Bulk Insert
        if (recordsToCreate.length > 0) {
            console.log(`${logPrefix} Creating ${recordsToCreate.length} new records via bulk API...`);
            const createResults = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "insert", options, recordsToCreate);
            console.log(`${logPrefix} Bulk create results received (${createResults.length}).`);
            createResults.forEach((res, index) => {
                if (!res.success) {
                    const recordIdentifier = recordsToCreate[index].Month__c || recordsToCreate[index].FY_Quarter__c;
                    console.error(`${logPrefix} Error creating record ${index + 1} (${recordIdentifier} ${recordsToCreate[index].Year__c}):`, JSON.stringify(res.errors));
                }
            });
        } else {
             console.log(`${logPrefix} No new records to create in this batch.`);
        }

        // Bulk Update
        if (recordsToUpdate.length > 0) {
            console.log(`${logPrefix} Updating ${recordsToUpdate.length} existing records via bulk API...`);
             const updateResults = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "update", options, recordsToUpdate);
             console.log(`${logPrefix} Bulk update results received (${updateResults.length}).`);
             updateResults.forEach((res, index) => {
                 if (!res.success) {
                    console.error(`${logPrefix} Error updating record ${index + 1} (ID: ${recordsToUpdate[index].Id}):`, JSON.stringify(res.errors));
                 }
             });
        } else {
             console.log(`${logPrefix} No existing records to update in this batch.`);
        }
    } catch (err) {
        console.error(`${logPrefix} Failed to save records to Salesforce using Bulk API: ${err.message}`, err.stack);
        throw new Error(`Salesforce save operation failed: ${err.message}`); // Propagate error
    }
}


// --- Callback Sending Function ---
// Sends the final status notification back to the specified callback URL
async function sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, status, message) {
    const logPrefix = `[Callback ${accountId}]`;
    // Truncate message for logging only
    const logMessage = message.length > 300 ? message.substring(0, 300) + '...' : message;
    console.log(`${logPrefix} Sending callback to ${callbackUrl}. Status: ${status}, Message snippet: ${logMessage}`);
    try {
        await axios.post(callbackUrl,
            {
                accountId: accountId,
                loggedinUserId: loggedinUserId,
                status: "Completed", // Status of the callback action itself
                processResult: status, // Overall process result ('Success', 'Failed', 'Partial Success')
                message: message // Full message/error list
            },
            {
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${accessToken}` // Authenticate callback if needed
                },
                timeout: 30000 // 30 second timeout for the callback request
            }
        );
        console.log(`${logPrefix} Callback sent successfully.`);
    } catch (error) {
        // Log detailed callback sending errors
        let errorMessage = error.message;
        if (error.response) {
            errorMessage = `Callback failed - Status: ${error.response.status}, Data: ${JSON.stringify(error.response.data)}, Msg: ${error.message}`;
        } else if (error.request) {
            errorMessage = `Callback failed - No response received. ${error.message}`;
        } else {
            errorMessage = `Callback failed - Error setting up request: ${error.message}`;
        }
        console.error(`${logPrefix} Failed to send callback: ${errorMessage}`);
        // Note: No retry logic implemented here by default
    }
}

// --- Utility Helper Functions ---

// Finds a value in the summaryRecordsMap array [{key:k, value:v}, ...]
function getValueByKey(recordsArray, searchKey) {
    if (!recordsArray || !Array.isArray(recordsArray) || !searchKey) return null;
    // Use case-insensitive search for robustness
    const keyLower = searchKey.toLowerCase();
    const record = recordsArray.find(item => item && typeof item.key === 'string' && item.key.toLowerCase() === keyLower);
    return record ? record.value : null;
}

// Transforms the raw quarterly AI output into the structure needed for Salesforce saving
function transformQuarterlyStructure(quarterlyAiOutput) {
    const result = {}; // Target: { year: { QX: { summaryDetails, summaryJson, count, startdate } } }

    // Validate the expected structure from the AI
    if (!quarterlyAiOutput?.yearlySummary?.[0]?.year ||
        !quarterlyAiOutput.yearlySummary[0]?.quarters?.[0]?.quarter ||
        !quarterlyAiOutput.yearlySummary[0]?.quarters?.[0]?.summary ||
        quarterlyAiOutput.yearlySummary[0]?.quarters?.[0]?.activityCount === undefined ||
        !quarterlyAiOutput.yearlySummary[0]?.quarters?.[0]?.startdate
       ) {
        console.warn("[Transform Quarterly] Invalid or incomplete structure received from quarterly AI:", JSON.stringify(quarterlyAiOutput).substring(0, 500));
        return result; // Return empty if validation fails
    }

    try {
        // Safely access nested data after validation
        const yearData = quarterlyAiOutput.yearlySummary[0];
        const year = yearData.year;
        result[year] = {};

        const quarterData = yearData.quarters[0];
        const quarter = quarterData.quarter;

        // Extract data for the Salesforce record fields
        const htmlSummary = quarterData.summary;
        const fullQuarterlyJson = JSON.stringify(quarterlyAiOutput); // Store the whole AI response
        const activityCount = quarterData.activityCount;
        const startDate = quarterData.startdate; // Expect YYYY-MM-DD

        // Assign to the result structure
        result[year][quarter] = {
            summaryDetails: htmlSummary,
            summaryJson: fullQuarterlyJson,
            count: activityCount,
            startdate: startDate
        };

    } catch (transformError) {
        console.error("[Transform Quarterly] Error during transformation:", transformError.stack);
        console.error("[Transform Quarterly] AI Output causing error (truncated):", JSON.stringify(quarterlyAiOutput).substring(0, 500));
        return {}; // Return empty on error
    }

    return result; // Return the structured data: e.g., { 2023: { Q1: { ... } } }
}

// =============================================================================
//                             End of Code
// =============================================================================
