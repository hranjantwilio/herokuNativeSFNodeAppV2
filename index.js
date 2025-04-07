/*
 * Enhanced Node.js Express application for generating Salesforce activity summaries using OpenAI Assistants.
 * Optimized for memory usage on platforms like Heroku using Bulk Query API Streaming.
 * VERSION: Full Code - Bulk Query Streaming Implementation
 *
 * Features:
 * - Salesforce Bulk Query API streaming to handle large datasets without loading all records into memory.
 * - Per-task OpenAI Assistant creation/deletion.
 * - Asynchronous processing with immediate acknowledgement.
 * - Salesforce integration (fetching activities via stream, saving summaries).
 * - OpenAI Assistants API V2 usage.
 * - Dynamic function schema loading (default or from request).
 * - Dynamic tool_choice to force specific function calls (monthly/quarterly).
 * - Conditional input method: Direct JSON in prompt (< threshold) or File Upload (>= threshold).
 * - Incremental processing: Generates & saves monthly summaries from stream batches, then aggregates & saves quarterly summaries individually.
 * - Robust error handling and callback mechanism.
 * - Temporary file management.
 * - Aggressive data truncation and sub-batching for memory control.
 */

// --- Dependencies ---
const express = require('express');
const jsforce = require('jsforce');
const dotenv = require('dotenv');
const { OpenAI, NotFoundError } = require("openai");
const fs = require("fs-extra");
const path = require("path");
const axios = require("axios");
const { Readable } = require('stream'); // For type checking if needed
const { pipeline } = require('stream/promises'); // For cleaner stream management if needed elsewhere

// --- Load Environment Variables ---
dotenv.config();

// --- Configuration Constants ---
const SF_LOGIN_URL = process.env.SF_LOGIN_URL;
const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";
const TIMELINE_SUMMARY_OBJECT_API_NAME = "Timeline_Summary__c"; // Salesforce object API name
const DIRECT_INPUT_THRESHOLD = 1500; // Max activities for direct JSON input
const PROMPT_LENGTH_THRESHOLD = 200000; // Max chars for prompt before switching to file upload
const TEMP_FILE_DIR = path.join(__dirname, 'temp_files'); // Directory for temporary files
const BULK_QUERY_BATCH_SIZE = 500; // Process records in sub-batches from the stream within a month
const DESCRIPTION_TRUNCATE_LENGTH = 1000; // Max chars for Description field passed to AI
const SUBJECT_TRUNCATE_LENGTH = 250;    // Max chars for Subject field passed to AI


// --- Environment Variable Validation ---
if (!SF_LOGIN_URL || !OPENAI_API_KEY) {
    console.error("FATAL ERROR: Missing required environment variables (SF_LOGIN_URL, OPENAI_API_KEY).");
    process.exit(1);
}

// --- Default OpenAI Function Schemas ---
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
    try {
        await fs.ensureDir(TEMP_FILE_DIR);
        console.log(`Temporary file directory ensured at: ${TEMP_FILE_DIR}`);
    } catch (err) {
        console.error(`FATAL: Could not create temporary directory ${TEMP_FILE_DIR}. Exiting.`, err);
        process.exit(1);
    }
});

// --- Main API Endpoint ---
app.post('/generatesummary', async (req, res) => {
    console.log("[Request] Received /generatesummary request");

    // --- Authorization ---
    const authHeader = req.headers["authorization"];
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
        console.warn("[Request] Unauthorized request: Missing or invalid Bearer token.");
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
        console.warn("[Request] Bad Request: Missing required parameters.");
        return res.status(400).send({ error: "Missing required parameters (accountId, callbackUrl, accessToken, queryText, userPrompt, userPromptQtr, loggedinUserId)" });
    }

     // Validate SOQL basic structure (simple check)
     if (!queryText.toLowerCase().includes('select') || !queryText.toLowerCase().includes('from') || !queryText.toLowerCase().includes('activitydate')) {
         console.warn("[Request] Bad Request: queryText seems invalid or missing ActivityDate field.");
         return res.status(400).send({ error: "queryText must be a valid SOQL query including the ActivityDate field." });
     }
     // Crucial for streaming logic: Ensure ORDER BY ActivityDate ASC is present
     if (!queryText.toLowerCase().includes('order by activitydate asc')) { // Check specifically for ASC
         console.warn("[Request] Bad Request: queryText MUST include 'ORDER BY ActivityDate ASC' for streaming logic to work correctly.");
         return res.status(400).send({ error: "queryText must include 'ORDER BY ActivityDate ASC' for streaming." });
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
            console.log("[Request] Using custom monthly function schema from request.");
        }
         if (qtrJSON) {
            quarterlyFuncSchema = JSON.parse(qtrJSON);
            if (!quarterlyFuncSchema || quarterlyFuncSchema.name !== 'generate_quarterly_activity_summary') {
                 throw new Error("Provided qtrJSON schema is invalid or missing the correct name property.");
            }
            console.log("[Request] Using custom quarterly function schema from request.");
        }
    } catch (e) {
        console.error("[Request] Failed to parse JSON input from request body:", e);
        return res.status(400).send({ error: `Invalid JSON provided in summaryMap, monthJSON, or qtrJSON. ${e.message}` });
    }

    // --- Ensure Schemas are Available ---
    if (!monthlyFuncSchema || !quarterlyFuncSchema) {
        console.error("[Request] FATAL: Default function schemas could not be loaded or found.");
        return res.status(500).send({ error: "Internal server error: Could not load function schemas."});
    }

    // --- Acknowledge Request (202 Accepted) ---
    res.status(202).json({ status: 'processing', message: 'Summary generation initiated via streaming. You will receive a callback.' });
    console.log(`[Request] Initiating summary processing for Account ID: ${accountId}`);

    // --- Start Asynchronous Processing ---
    processSummary( // Call the Bulk Query Streaming version
        accountId,
        accessToken,
        callbackUrl,
        userPrompt,
        userPromptQtr,
        queryText, // Pass the SOQL query
        summaryRecordsMap,
        loggedinUserId,
        monthlyFuncSchema,
        quarterlyFuncSchema
    ).catch(async (error) => {
        console.error(`[Request] Unhandled error during background processing setup for ${accountId}:`, error);
        try {
            await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Unhandled processing setup error: ${error.message}`);
        } catch (callbackError) {
            console.error(`[Request] Failed to send error callback after unhandled setup exception for ${accountId}:`, callbackError);
        }
    });
});

// --- Helper Function to Get Quarter from Month Index ---
function getQuarterFromMonthIndex(monthIndex) {
    if (monthIndex >= 0 && monthIndex <= 2) return 'Q1';
    if (monthIndex >= 3 && monthIndex <= 5) return 'Q2';
    if (monthIndex >= 6 && monthIndex <= 8) return 'Q3';
    if (monthIndex >= 9 && monthIndex <= 11) return 'Q4';
    return 'Unknown';
}

// --- Asynchronous Summary Processing Logic (Refactored for Bulk Query Streaming) ---
async function processSummary(
    accountId,
    accessToken,
    callbackUrl,
    userPromptMonthlyTemplate,
    userPromptQuarterlyTemplate,
    queryText, // SOQL query
    summaryRecordsMap,
    loggedinUserId,
    monthlyFuncSchema,
    quarterlyFuncSchema
) {
    const logPrefix = `[Process ${accountId}]`;
    console.log(`${logPrefix} Starting processSummary (Bulk Query Streaming)...`);
    const conn = new jsforce.Connection({
        instanceUrl: SF_LOGIN_URL,
        accessToken: accessToken,
        maxRequest: 10,
        // Optional: Configure Bulk API options if needed, e.g., { pollTimeout: 120000 }
    });

    const quarterlyInputs = {}; // Stores AI output from monthly summaries for quarterly aggregation
    const monthMap = { january: 0, february: 1, march: 2, april: 3, may: 4, june: 5, july: 6, august: 7, september: 8, october: 9, november: 10, december: 11 };
    let overallStatus = "Success";
    let failureMessages = [];
    let recordCount = 0;
    let currentYear = null;
    let currentMonth = null;
    let currentMonthActivities = []; // Batch of activities for the current month/sub-batch
    let bulkQueryJob = null; // Reference to the streaming job

    // --- Helper Function to process a completed month's batch of activities ---
    async function processAndSaveMonthBatch(year, month, activities) {
        const activityCount = activities.length;
        if (activityCount === 0) return; // Should not happen if called correctly, but safe check

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
            const assistantName = `Monthly Summarizer ${accountId} ${year}-${month}-${Date.now()}`;
            console.log(`${batchLogPrefix}   Creating Assistant: ${assistantName}`);
            monthlyAssistant = await openai.beta.assistants.create({
               name: assistantName,
               instructions: "You are an AI assistant specialized in analyzing raw Salesforce activity data for a single month (fields like Description/Subject may be truncated) and generating structured JSON summaries using the provided function 'generate_monthly_activity_summary'. Apply sub-theme segmentation. Focus on key themes, tone, and actions.",
               tools: [{ type: "file_search" }, { type: "function", "function": monthlyFuncSchema }],
               model: OPENAI_MODEL,
            });
            console.log(`${batchLogPrefix}   Created Assistant ID: ${monthlyAssistant.id}`);

            // console.log(`${batchLogPrefix}   Memory before generateSummary:`, process.memoryUsage()); // Uncomment for debug

            monthlyAiOutput = await generateSummary(activities, openai, monthlyAssistant.id, userPromptMonthly, monthlyFuncSchema, accountId);
            console.log(`${batchLogPrefix}   Generated monthly summary.`);

            // console.log(`${batchLogPrefix}   Memory after generateSummary:`, process.memoryUsage()); // Uncomment for debug

            const monthlyForSalesforce = {
               [year]: { [month]: {
                   summary: JSON.stringify(monthlyAiOutput), // Full AI JSON
                   summaryDetails: monthlyAiOutput?.summary || '', // HTML part
                   count: activityCount, // Count for this specific batch
                   startdate: startDateStr
               }}
            };

            console.log(`${batchLogPrefix}   Saving monthly summary to Salesforce...`);
            await createTimileSummarySalesforceRecords(conn, monthlyForSalesforce, accountId, 'Monthly', summaryRecordsMap);
            console.log(`${batchLogPrefix}   Saved monthly summary.`);

            // Store the AI output for later quarterly aggregation
            const quarter = getQuarterFromMonthIndex(monthIndex);
            const quarterKey = `${year}-${quarter}`;
            if (!quarterlyInputs[quarterKey]) {
                quarterlyInputs[quarterKey] = { monthSummaries: [], year: parseInt(year), quarter: quarter };
            }
            quarterlyInputs[quarterKey].monthSummaries.push(monthlyAiOutput);

        } catch (monthError) {
            console.error(`${batchLogPrefix} Error processing batch:`, monthError.stack);
            overallStatus = "Failed"; // Mark process as failed if any batch fails
            failureMessages.push(`Failed processing ${month} ${year}: ${monthError.message}`);
        } finally {
            // Cleanup Assistant regardless of success/failure
            if (monthlyAssistant?.id) {
                try {
                    console.log(`${batchLogPrefix}   Deleting monthly assistant ${monthlyAssistant.id}`);
                    await openai.beta.assistants.del(monthlyAssistant.id);
                    console.log(`${batchLogPrefix}   Deleted monthly assistant ${monthlyAssistant.id}`);
                } catch (delError) {
                    // Log deletion error but don't fail the process for this
                    console.warn(`${batchLogPrefix}   Could not delete monthly assistant ${monthlyAssistant.id}:`, delError.message || delError);
                 }
            }
            // Explicitly nullify large objects to potentially help GC
            activities = null;
            monthlyAiOutput = null;
            // Optional: Trigger Garbage Collection more aggressively if needed
            // if (global.gc) { console.log(`${batchLogPrefix} Triggering GC`); global.gc(); }
        }
    } // --- End of processAndSaveMonthBatch ---

    try {
        // 1. Execute Bulk Query and Stream Results
        console.log(`${logPrefix} Starting Bulk API Query stream... Query: ${queryText.substring(0,150)}...`);
        // Ensure the connection is ready before starting the query if needed (usually handled by jsforce)
        bulkQueryJob = conn.bulk.query(queryText);
        let streamPaused = false; // Track pause state

        // --- Stream Record Event Handler ---
        bulkQueryJob.on('record', async (record) => {
             recordCount++;
             // Only process if the stream isn't manually paused for batch processing
             if (!streamPaused) {
                 // Basic validation of the incoming record
                 if (!record.ActivityDate) {
                     console.warn(`[Stream Record ${recordCount} ${accountId}] Skipping activity (ID: ${record.Id || 'Unknown'}) - missing ActivityDate.`);
                     return;
                 }

                 try {
                     const date = new Date(record.ActivityDate);
                     if (isNaN(date.getTime())) {
                         console.warn(`[Stream Record ${recordCount} ${accountId}] Skipping activity (ID: ${record.Id || 'Unknown'}) - invalid ActivityDate: ${record.ActivityDate}`);
                         return;
                     }

                     const year = date.getUTCFullYear();
                     const month = date.toLocaleString('en-US', { month: 'long', timeZone: 'UTC' });

                     // --- Month Change Detection ---
                     if (year !== currentYear || month !== currentMonth) {
                         // If we have accumulated activities from the *previous* month, process them now.
                         if (currentMonthActivities.length > 0) {
                             streamPaused = true; // Set flag BEFORE pausing
                             console.log(`[Stream ${accountId}] Pausing stream to process completed month: ${currentMonth} ${currentYear} (${currentMonthActivities.length} records)`);
                             bulkQueryJob.pause();

                             // Process the completed batch
                             await processAndSaveMonthBatch(currentYear, currentMonth, currentMonthActivities);

                             currentMonthActivities = []; // Clear the processed batch
                             // Resume stream only if the job is still in a valid state
                             if (bulkQueryJob && bulkQueryJob.job && bulkQueryJob.job.state !== 'Closed' && bulkQueryJob.job.state !== 'Aborted') {
                                console.log(`[Stream ${accountId}] Resuming stream after processing ${currentMonth} ${currentYear}`);
                                streamPaused = false; // Reset flag AFTER resuming
                                bulkQueryJob.resume();
                             } else {
                                console.log(`[Stream ${accountId}] Stream job is closed or aborted, not resuming after processing ${currentMonth} ${currentYear}.`);
                                streamPaused = false; // Ensure flag is reset even if not resuming
                             }
                         }
                         // Update the current processing state to the new month/year
                         currentYear = year;
                         currentMonth = month;
                     }

                     // Add essential, TRUNCATED data to the current batch
                     currentMonthActivities.push({
                        Id: record.Id,
                        Description: record.Description?.substring(0, DESCRIPTION_TRUNCATE_LENGTH) || null,
                        Subject: record.Subject?.substring(0, SUBJECT_TRUNCATE_LENGTH) || null,
                        ActivityDate: record.ActivityDate // Keep original date format
                        // Add any other essential fields needed by AI here, potentially truncated
                     });

                     // --- Sub-Batch Processing within a Month ---
                     if (currentMonthActivities.length >= BULK_QUERY_BATCH_SIZE) {
                          streamPaused = true; // Set flag BEFORE pausing
                          console.log(`[Stream ${accountId}] Pausing stream to process sub-batch for ${currentMonth} ${currentYear} (size ${currentMonthActivities.length})`);
                          bulkQueryJob.pause();

                          // Process the sub-batch
                          await processAndSaveMonthBatch(currentYear, currentMonth, currentMonthActivities);

                          currentMonthActivities = []; // Clear the processed sub-batch
                          // Resume stream only if the job is still valid
                          if (bulkQueryJob && bulkQueryJob.job && bulkQueryJob.job.state !== 'Closed' && bulkQueryJob.job.state !== 'Aborted') {
                            console.log(`[Stream ${accountId}] Resuming stream after processing sub-batch for ${currentMonth} ${currentYear}`);
                            streamPaused = false; // Reset flag AFTER resuming
                            bulkQueryJob.resume();
                          } else {
                             console.log(`[Stream ${accountId}] Stream job is closed or aborted, not resuming after processing sub-batch for ${currentMonth} ${currentYear}.`);
                             streamPaused = false; // Ensure flag is reset
                          }
                     }

                 } catch (recordError) {
                      console.error(`[Stream Record ${recordCount} ${accountId}] Error processing record (ID: ${record.Id || 'Unknown'}):`, recordError.stack);
                      // Optionally mark overallStatus as failed here, or just log and continue
                      // overallStatus = "Failed";
                      // failureMessages.push(`Error processing record ${record.Id}: ${recordError.message}`);
                 }
             } else {
                  // This log can be very verbose, uncomment only if needed for deep debugging pause/resume issues
                  // console.log(`[Stream Record ${recordCount} ${accountId}] Skipped processing because stream is paused.`);
             }
        }); // --- End of 'record' Handler ---

        // --- Promise to Wait for Stream Completion ---
        await new Promise((resolve, reject) => {
            bulkQueryJob.on('error', (err) => {
                console.error(`${logPrefix} Error during Bulk API Query stream:`, err);
                overallStatus = "Failed";
                failureMessages.push(`Salesforce query stream error: ${err.message}`);
                // Ensure stream processing stops if possible (jsforce might handle this internally on error)
                reject(err); // Reject the promise
            });

            bulkQueryJob.on('end', async () => {
                console.log(`${logPrefix} Bulk API Query stream finished. Total records received: ${recordCount}`);
                try {
                    // Process the very last batch of activities accumulated
                    if (currentMonthActivities.length > 0) {
                        console.log(`${logPrefix} Processing final batch for ${currentMonth} ${currentYear} (${currentMonthActivities.length} records)`);
                        await processAndSaveMonthBatch(currentYear, currentMonth, currentMonthActivities);
                        currentMonthActivities = []; // Clear the final batch
                    }
                    console.log(`${logPrefix} Finished processing all monthly data from stream.`);
                    resolve(); // Resolve the promise indicating stream processing is complete
                } catch (finalBatchError) {
                    console.error(`${logPrefix} Error processing final month batch for ${currentMonth} ${currentYear}:`, finalBatchError.stack);
                    overallStatus = "Failed";
                    failureMessages.push(`Failed processing final batch ${currentMonth} ${currentYear}: ${finalBatchError.message}`);
                    reject(finalBatchError); // Reject if the final batch fails
                }
            });
        }); // --- End of Stream Promise ---

        // --- Optional GC & memory log after all stream processing ---
        // if (global.gc) { global.gc(); }
        // console.log(`${logPrefix} Memory usage after stream processing:`, process.memoryUsage());

        // 2. Process Quarters Incrementally (Using aggregated `quarterlyInputs`)
        console.log(`${logPrefix} Processing ${Object.keys(quarterlyInputs).length} quarters based on streamed data...`);
        for (const quarterKey in quarterlyInputs) {
            // --- Start of Quarterly Loop ---
            const quarterlyLogPrefix = `${logPrefix} -> [Quarter ${quarterKey}]`;
            let { monthSummaries, year, quarter } = quarterlyInputs[quarterKey];
            const numMonthlySummaries = monthSummaries?.length || 0;
            console.log(`${quarterlyLogPrefix} Generating summary using ${numMonthlySummaries} monthly summaries...`);

            if (numMonthlySummaries === 0) {
                console.warn(`${quarterlyLogPrefix} Skipping as it has no associated monthly summaries.`);
                continue;
            }

            // Prepare the JSON input string for the quarterly prompt
            const quarterlyInputDataString = JSON.stringify(monthSummaries, null, 2); // Pretty print for AI readability
            const userPromptQuarterly = `${userPromptQuarterlyTemplate.replace('{{Quarter}}', quarter).replace('{{Year}}', year)}\n\nAggregate the following monthly summary data provided below for ${quarterKey}:\n\`\`\`json\n${quarterlyInputDataString}\n\`\`\``;

            // Check if quarterly input itself is too large (could potentially use file upload here too if needed)
            if(quarterlyInputDataString.length > PROMPT_LENGTH_THRESHOLD * 2) { // Example threshold check
                console.warn(`${quarterlyLogPrefix} Combined monthly summaries JSON (${quarterlyInputDataString.length} chars) is very large. Consider optimizing monthly output or using file input for quarterly stage.`);
                // Potentially skip or try file upload for quarterly stage if this becomes an issue
            }

            let quarterlyAiOutput = null;
            let quarterlyAssistant = null;

            try {
                const assistantName = `Quarterly Summarizer ${accountId} ${quarterKey}-${Date.now()}`;
                console.log(`${quarterlyLogPrefix}   Creating Assistant: ${assistantName}`);
                quarterlyAssistant = await openai.beta.assistants.create({
                     name: assistantName,
                     instructions: "You are an AI assistant specialized in aggregating pre-summarized monthly Salesforce activity data (provided as JSON in the prompt) into a structured quarterly JSON summary for a specific quarter using the provided function 'generate_quarterly_activity_summary'. Consolidate insights and activity lists accurately.",
                     tools: [{ type: "file_search" }, { type: "function", "function": quarterlyFuncSchema }],
                     model: OPENAI_MODEL,
                });
                console.log(`${quarterlyLogPrefix}   Created Assistant ID: ${quarterlyAssistant.id}`);

                // console.log(`${quarterlyLogPrefix}   Memory before quarterly generateSummary:`, process.memoryUsage()); // Uncomment for debug

                // Call generateSummary with null activities (data is in prompt)
                quarterlyAiOutput = await generateSummary(null, openai, quarterlyAssistant.id, userPromptQuarterly, quarterlyFuncSchema, accountId);
                console.log(`${quarterlyLogPrefix}   Generated quarterly summary AI output.`);

                // console.log(`${quarterlyLogPrefix}   Memory after quarterly generateSummary:`, process.memoryUsage()); // Uncomment for debug

                // Transform the raw AI output to the Salesforce save structure
                const transformedQuarter = transformQuarterlyStructure(quarterlyAiOutput);

                // Validate transformation and save
                if (transformedQuarter && transformedQuarter[year] && transformedQuarter[year][quarter]) {
                    const quarterlyForSalesforce = { [year]: { [quarter]: transformedQuarter[year][quarter] } };
                    console.log(`${quarterlyLogPrefix}   Saving quarterly summary to Salesforce...`);
                    await createTimileSummarySalesforceRecords(conn, quarterlyForSalesforce, accountId, 'Quarterly', summaryRecordsMap);
                    console.log(`${quarterlyLogPrefix}   Saved quarterly summary.`);
                } else {
                     console.warn(`${quarterlyLogPrefix} Quarterly summary generated by AI but transform/validation failed. Skipping save.`);
                     if (overallStatus !== "Failed") { overallStatus = "Partial Success"; } // Mark partial success if not already failed
                     failureMessages.push(`Failed to transform/validate AI output for ${quarterKey}.`);
                }

            } catch (quarterlyError) {
                console.error(`${quarterlyLogPrefix} Failed to generate or save quarterly summary:`, quarterlyError.stack);
                overallStatus = "Failed"; // Mark overall as failed
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
                 // Release memory references for this quarter's data
                 if (quarterlyInputs[quarterKey]) quarterlyInputs[quarterKey].monthSummaries = null;
                 quarterlyAiOutput = null;
                 monthSummaries = null; // Release local reference
                 // Optional GC & memory log
                 // if (global.gc) { console.log(`${quarterlyLogPrefix} Triggering GC`); global.gc(); }
           }
             // --- End of Quarterly Loop ---
        }

        // 3. Send Final Callback
        console.log(`${logPrefix} Process completed.`);
        let finalMessage = overallStatus === "Success" ? "Summary Processed Successfully" : `Processing finished with issues: ${failureMessages.join('; ')}`;
        // Truncate potentially long failure message lists
        finalMessage = finalMessage.length > 1000 ? finalMessage.substring(0, 997) + "..." : finalMessage;
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, overallStatus, finalMessage);

    } catch (error) {
        // Catch errors from stream setup, stream promise rejection, or other unhandled exceptions
        console.error(`${logPrefix} Critical error during summary processing:`, error.stack);
        // Ensure a failure callback is sent
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Critical processing error: ${error.message}`);
    } finally {
         console.log(`${logPrefix} processSummary finished execution.`);
         // Optional final GC & memory log
        //  if (global.gc) { console.log(`${logPrefix} Triggering final GC.`); global.gc(); }
        //  console.log(`${logPrefix} Final memory usage:`, process.memoryUsage());
    }
}


// --- OpenAI Summary Generation Function (Includes conditional input) ---
async function generateSummary(
    activities, // Array of raw activities OR null
    openaiClient,
    assistantId,
    userPrompt,
    functionSchema,
    accountId = 'N/A' // Add accountId for logging context
) {
    let fileId = null;
    let thread = null;
    let tempFilePath = null;
    let inputMethod = "prompt";
    // Dynamically create log prefix once thread ID is available
    let logPrefix = `[AI ${accountId} Thread New]`;

    try {
        thread = await openaiClient.beta.threads.create();
        logPrefix = `[AI ${accountId} Thread ${thread.id}]`; // Update log prefix
        console.log(`${logPrefix} Created for Assistant ${assistantId}`);

        let finalUserPrompt = userPrompt;
        let messageAttachments = [];

        // Determine input method based on activities provided
        if (activities && Array.isArray(activities) && activities.length > 0) {
            let potentialFullPrompt;
            let activitiesJsonString;

            try {
                 // Stringify without pretty print for initial length check (saves memory)
                 activitiesJsonString = JSON.stringify(activities);
                 potentialFullPrompt = `${userPrompt}\n\nActivity data:\n\`\`\`json\n${activitiesJsonString}\n\`\`\``;
                 console.log(`${logPrefix} Potential prompt length with direct JSON: ${potentialFullPrompt.length} chars.`);
            } catch(stringifyError) {
                console.error(`${logPrefix} Error stringifying activities for length check:`, stringifyError);
                // Don't proceed if we can't even stringify
                throw new Error("Failed to stringify activity data for processing.");
            }

            // Check thresholds for direct input vs. file upload
            if (potentialFullPrompt.length < PROMPT_LENGTH_THRESHOLD && activities.length < DIRECT_INPUT_THRESHOLD) {
                inputMethod = "direct JSON";
                // Include pretty-printed JSON in the final prompt for better AI readability if it fits
                finalUserPrompt = `${userPrompt}\n\nActivity data:\n\`\`\`json\n${JSON.stringify(activities, null, 2)}\n\`\`\``;
                console.log(`${logPrefix} Using direct JSON input (${activities.length} activities).`);
            } else {
                // --- File Upload Method ---
                inputMethod = "file upload";
                console.log(`${logPrefix} Using file upload (Activities: ${activities.length} >= ${DIRECT_INPUT_THRESHOLD} or Prompt Length: ${potentialFullPrompt.length} >= ${PROMPT_LENGTH_THRESHOLD}).`);
                // Use the original base userPrompt (without embedded JSON)
                finalUserPrompt = userPrompt;

                // Convert activities to Plain Text format for file_search compatibility
                // Use the already potentially truncated activity data passed into this function
                let activitiesText = activities.map((activity, index) => {
                    let desc = activity.Description || 'No Description'; // Handle null
                    let subj = activity.Subject || 'No Subject';     // Handle null
                    return `Activity ${index + 1} (ID: ${activity.Id || 'N/A'}):\n  ActivityDate: ${activity.ActivityDate || 'N/A'}\n  Subject: ${subj}\n  Description: ${desc}`;
                }).join('\n\n---\n\n'); // Use a clear separator

                const timestamp = new Date().toISOString().replace(/[:.-]/g, "_");
                const filename = `activities_${accountId}_${timestamp}_${thread.id}.txt`; // Use .txt extension
                tempFilePath = path.join(TEMP_FILE_DIR, filename); // Use configured temp directory

                // Write the text file asynchronously
                await fs.writeFile(tempFilePath, activitiesText);
                console.log(`${logPrefix} Temporary text file generated: ${tempFilePath}`);

                // Upload the file using a stream for potentially large text files
                const fileStream = fs.createReadStream(tempFilePath);
                const uploadResponse = await openaiClient.files.create({
                    file: fileStream,
                    purpose: "assistants", // Must be 'assistants' for V2 API
                });
                fileId = uploadResponse.id;
                console.log(`${logPrefix} File uploaded to OpenAI: ${fileId}`);

                // Attach the file to the message using the 'file_search' tool type
                messageAttachments.push({ file_id: fileId, tools: [{ type: "file_search" }] });
                console.log(`${logPrefix} Attaching file ${fileId} with file_search tool.`);
                // Add instruction for AI to use the file
                finalUserPrompt += "\n\nPlease analyze the activity data provided in the attached file.";
            }
        } else {
             // Case where no activities are passed (e.g., quarterly summary using prompt data)
             console.log(`${logPrefix} No activities array provided or array is empty. Using prompt content as is.`);
        }

        // Create the message payload
        const messagePayload = { role: "user", content: finalUserPrompt };
        if (messageAttachments.length > 0) {
            messagePayload.attachments = messageAttachments;
        }

        // Add the message to the thread
        const message = await openaiClient.beta.threads.messages.create(thread.id, messagePayload);
        console.log(`${logPrefix} Message added (using ${inputMethod}). ID: ${message.id}`);

        // Run the assistant and poll for completion, forcing the specific function
        console.log(`${logPrefix} Starting run, forcing function: ${functionSchema.name}`);
        const run = await openaiClient.beta.threads.runs.createAndPoll(thread.id, {
            assistant_id: assistantId,
            // Pass the specific function schema the Assistant should use for this run
            tools: [{ type: "function", function: functionSchema }],
            // Explicitly force the assistant to call THIS function
            tool_choice: { type: "function", function: { name: functionSchema.name } },
            // Optional: Add a timeout for the poll if needed
            // pollIntervalMs: 5000, // Default polling interval
            // timeout: 10 * 60 * 1000, // Example: 10 minute timeout for the entire poll
        });
        console.log(`${logPrefix} Run status: ${run.status}`);

        // --- Process Run Outcome ---
        if (run.status === 'requires_action') {
            const toolCalls = run.required_action?.submit_tool_outputs?.tool_calls;
            // Validate the tool call structure
            if (!toolCalls || toolCalls.length === 0 || !toolCalls[0]?.function?.arguments) {
                 console.error(`${logPrefix} Run requires action, but tool call data is missing or invalid.`, run.required_action);
                 throw new Error("Function call was expected but not provided correctly by the Assistant.");
             }
             const toolCall = toolCalls[0]; // Assuming one function call per run based on tool_choice

             // Verify the correct function was called
             if (toolCall.function.name !== functionSchema.name) {
                  console.error(`${logPrefix} Assistant called the wrong function. Expected: ${functionSchema.name}, Got: ${toolCall.function.name}`);
                  throw new Error(`Assistant called the wrong function: ${toolCall.function.name}`);
             }

             const rawArgs = toolCall.function.arguments;
             console.log(`${logPrefix} Function call arguments received for ${toolCall.function.name}. Length: ${rawArgs.length}`);
             try {
                 // Parse the JSON arguments returned by the AI function call
                 const summaryObj = JSON.parse(rawArgs);
                 console.log(`${logPrefix} Successfully parsed function arguments.`);
                 // Return the parsed object - no need to submit tool outputs back for this use case
                 return summaryObj;
             } catch (parseError) {
                 console.error(`${logPrefix} Failed to parse function call arguments JSON:`, parseError);
                 console.error(`${logPrefix} Raw arguments received (first 500 chars):`, rawArgs.substring(0, 500));
                 throw new Error(`Failed to parse function call arguments from AI: ${parseError.message}`);
             }
         } else if (run.status === 'completed') {
              // This is unexpected when tool_choice forces a function call
              console.warn(`${logPrefix} Run completed without requiring function call action, despite tool_choice forcing ${functionSchema.name}. Check Assistant logic/prompt.`);
              // Try to get the last assistant message for debugging clues
              try {
                const messages = await openaiClient.beta.threads.messages.list(run.thread_id, { order: 'desc', limit: 1 });
                const lastMessageContent = messages.data[0]?.content[0]?.text?.value || "No text content found.";
                console.warn(`${logPrefix} Last message content from Assistant: ${lastMessageContent.substring(0, 500)}...`);
              } catch (msgError) {
                  console.warn(`${logPrefix} Could not retrieve last message for completed run: ${msgError.message}`);
              }
              throw new Error(`Assistant run completed without making the required function call to ${functionSchema.name}.`);
         } else {
             // Handle other terminal statuses: 'failed', 'cancelled', 'expired'
             console.error(`${logPrefix} Run failed or ended unexpectedly. Status: ${run.status}`, run.last_error || run.incomplete_details);
             const errorMessage = run.last_error ? `${run.last_error.code}: ${run.last_error.message}` : (run.incomplete_details?.reason || 'Unknown error');
             throw new Error(`Assistant run failed. Status: ${run.status}. Error: ${errorMessage}`);
         }

    } catch (error) {
        // Log any errors caught during thread creation, message sending, run execution, etc.
        // Use the logPrefix established after thread creation if possible
        const currentLogPrefix = `[AI ${accountId} Thread ${thread?.id || 'N/A'}]`;
        console.error(`${currentLogPrefix} Error in generateSummary: ${error.message}`);
        console.error(error.stack); // Log stack trace for better debugging
        throw error; // Re-throw the error to be handled by the caller (processSummary)
    } finally {
        // --- Cleanup Resources ---
        const currentLogPrefix = `[AI ${accountId} Thread ${thread?.id || 'Cleanup'}]`;
        // Delete the temporary local file if it was created
        if (tempFilePath) {
            try {
                await fs.unlink(tempFilePath);
                console.log(`${currentLogPrefix} Deleted temporary file: ${tempFilePath}`);
            } catch (unlinkError) {
                // Log error but don't fail the entire operation just for cleanup failure
                console.error(`${currentLogPrefix} Error deleting temporary file ${tempFilePath}:`, unlinkError);
            }
        }
        // Delete the file uploaded to OpenAI if it was created
        if (fileId) {
            try {
                await openaiClient.files.del(fileId);
                console.log(`${currentLogPrefix} Deleted OpenAI file: ${fileId}`);
            } catch (deleteError) {
                 // Ignore 404 errors (file already deleted), log others
                 if (!(deleteError instanceof NotFoundError || deleteError?.status === 404)) {
                    console.error(`${currentLogPrefix} Error deleting OpenAI file ${fileId}:`, deleteError.message || deleteError);
                 } else {
                     console.log(`${currentLogPrefix} OpenAI file ${fileId} already deleted or not found.`);
                 }
            }
        }
        // Optional: Delete the thread itself to manage OpenAI resource usage strictly
        // if (thread?.id) {
        //     try {
        //         await openaiClient.beta.threads.del(thread.id);
        //         console.log(`${currentLogPrefix} Deleted thread.`);
        //      } catch(e){
        //          console.warn(`${currentLogPrefix} Failed to delete thread ${thread.id}: ${e.message}`);
        //      }
        // }
    }
}

// --- Salesforce Record Creation/Update Function (Bulk API) ---
async function createTimileSummarySalesforceRecords(conn, summaries, parentId, summaryCategory, summaryRecordsMap) {
    const logPrefix = `[SF Save ${parentId} ${summaryCategory}]`;
    console.log(`${logPrefix} Preparing to save summaries...`);
    let recordsToCreate = [];
    let recordsToUpdate = [];

    // Iterate through the summaries structure { year: { periodKey: { summaryJson/summary, summaryDetails, count, startdate } } }
    for (const year in summaries) {
        for (const periodKey in summaries[year]) { // periodKey is 'MonthName' or 'Q1', 'Q2' etc.
            const summaryData = summaries[year][periodKey];

            // Extract data, using summaryJson/summary for the full AI response
            let summaryJsonString = summaryData.summaryJson || summaryData.summary;
            let summaryDetailsHtml = summaryData.summaryDetails || ''; // Extracted HTML part
            let startDate = summaryData.startdate; // Should be YYYY-MM-DD
            let count = summaryData.count;

             // Fallback: Try to extract HTML from the full JSON if details field is empty/missing
             if (!summaryDetailsHtml && summaryJsonString) {
                try {
                    const parsedJson = JSON.parse(summaryJsonString);
                    // Check common paths for the HTML summary based on monthly/quarterly structures
                    summaryDetailsHtml = parsedJson.summary || // Check top level (monthly schema)
                                        parsedJson?.yearlySummary?.[0]?.quarters?.[0]?.summary || // Check quarterly schema path
                                        ''; // Default to empty if not found
                } catch (e) {
                    console.warn(`${logPrefix} Could not parse 'summaryJsonString' for ${periodKey} ${year} to extract HTML details. HTML field might be empty.`);
                 }
            }

            // Determine SF fields based on category
            let fyQuarterValue = (summaryCategory === 'Quarterly') ? periodKey : '';
            let monthValue = (summaryCategory === 'Monthly') ? periodKey : '';
            // Create a consistent key for looking up existing records in summaryRecordsMap
            let shortMonth = monthValue ? monthValue.substring(0, 3) : '';
            let summaryMapKey = (summaryCategory === 'Quarterly') ? `${fyQuarterValue} ${year}` : `${shortMonth} ${year}`;
            let existingRecordId = getValueByKey(summaryRecordsMap, summaryMapKey);

            // Prepare Salesforce Record Payload (Ensure API names match your custom object)
            const recordPayload = {
                // Parent_Id__c: parentId, // Lookup to parent record (e.g., Account) - Use standard Account lookup below if applicable
                Month__c: monthValue || null, // Text field for month name
                Year__c: String(year), // Text or Number field for year
                Summary_Category__c: summaryCategory, // Picklist ('Monthly', 'Quarterly')
                // Use substring to prevent exceeding Salesforce field limits (e.g., 131072)
                Summary__c: summaryJsonString ? summaryJsonString.substring(0, 131070) : null, // Long Text Area (Full AI JSON)
                Summary_Details__c: summaryDetailsHtml ? summaryDetailsHtml.substring(0, 131070) : null, // Rich Text Area (HTML Summary)
                FY_Quarter__c: fyQuarterValue || null, // Text field for quarter (e.g., 'Q1')
                Month_Date__c: startDate, // Date field for the start of the period
                Number_of_Records__c: count, // Number field for activity count in this batch/period
                Account__c: parentId // Standard Lookup field to Account (assuming Parent_Id__c isn't used) - ADJUST IF NEEDED
            };

             // Basic validation before adding to queue
             if (!recordPayload.Account__c || !recordPayload.Summary_Category__c || !recordPayload.Year__c) {
                 console.warn(`${logPrefix} Skipping record for ${summaryMapKey} - missing Account ID, Category, or Year.`);
                 continue;
             }

            // Add to appropriate list for bulk operation
            if (existingRecordId) {
                console.log(`${logPrefix}   Queueing update for ${summaryMapKey} (ID: ${existingRecordId})`);
                recordsToUpdate.push({ Id: existingRecordId, ...recordPayload });
            } else {
                console.log(`${logPrefix}   Queueing create for ${summaryMapKey}`);
                recordsToCreate.push(recordPayload);
            }
        }
    }

    // --- Perform Bulk DML Operations ---
    try {
        // Use allOrNone=false to allow partial success, essential for batch processing
        const options = { allOrNone: false };

        if (recordsToCreate.length > 0) {
            console.log(`${logPrefix} Creating ${recordsToCreate.length} new records via bulk API...`);
            // Use bulk API V1 load for simplicity here, consider V2 if needed for very large scale
            const createResults = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "insert", options, recordsToCreate);
            console.log(`${logPrefix} Bulk create results received (${createResults.length}).`);
            // Log errors for failed records
            createResults.forEach((res, index) => {
                if (!res.success) {
                    const recordIdentifier = recordsToCreate[index].Month__c || recordsToCreate[index].FY_Quarter__c;
                    console.error(`${logPrefix} Error creating record ${index + 1} (${recordIdentifier} ${recordsToCreate[index].Year__c}):`, JSON.stringify(res.errors));
                    // Consider adding failed record details to failureMessages for callback
                }
            });
        } else {
             console.log(`${logPrefix} No new records to create in this batch.`);
        }

        if (recordsToUpdate.length > 0) {
            console.log(`${logPrefix} Updating ${recordsToUpdate.length} existing records via bulk API...`);
             const updateResults = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "update", options, recordsToUpdate);
             console.log(`${logPrefix} Bulk update results received (${updateResults.length}).`);
             // Log errors for failed updates
             updateResults.forEach((res, index) => {
                 if (!res.success) {
                    console.error(`${logPrefix} Error updating record ${index + 1} (ID: ${recordsToUpdate[index].Id}):`, JSON.stringify(res.errors));
                    // Consider adding failed record details to failureMessages
                 }
             });
        } else {
             console.log(`${logPrefix} No existing records to update in this batch.`);
        }
    } catch (err) {
        console.error(`${logPrefix} Failed to save records to Salesforce using Bulk API: ${err.message}`, err.stack);
        // Throw error to be caught by the calling function (processAndSaveMonthBatch or quarterly loop)
        throw new Error(`Salesforce save operation failed: ${err.message}`);
    }
}

// --- Callback Sending Function ---
async function sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, status, message) {
    const logPrefix = `[Callback ${accountId}]`;
    // Truncate long messages before logging
    const logMessage = message.length > 300 ? message.substring(0, 300) + '...' : message;
    console.log(`${logPrefix} Sending callback to ${callbackUrl}. Status: ${status}, Message snippet: ${logMessage}`);
    try {
        await axios.post(callbackUrl,
            {
                // Payload structure expected by the Salesforce callback receiver
                accountId: accountId,
                loggedinUserId: loggedinUserId,
                status: "Completed", // Status of the *callback sending action* itself
                processResult: status, // Overall result ('Success', 'Failed', 'Partial Success') of the summary generation
                message: message // Detailed message or error list (potentially truncated in logs only)
            },
            {
                headers: {
                    "Content-Type": "application/json",
                    // Use the provided Salesforce access token for authenticating the callback request
                    "Authorization": `Bearer ${accessToken}`
                },
                timeout: 30000 // Increased timeout (30 seconds) for potentially slow callback receivers
            }
        );
        console.log(`${logPrefix} Callback sent successfully.`);
    } catch (error) {
        let errorMessage = error.message;
        if (error.response) {
            // Include response details if available (status code, data)
            errorMessage = `Callback failed - Status: ${error.response.status}, Data: ${JSON.stringify(error.response.data)}, Message: ${error.message}`;
        } else if (error.request) {
            // The request was made but no response was received
            errorMessage = `Callback failed - No response received from callback URL. ${error.message}`;
        } else {
            // Something happened in setting up the request that triggered an Error
            errorMessage = `Callback failed - Error setting up request: ${error.message}`;
        }
        console.error(`${logPrefix} Failed to send callback: ${errorMessage}`);
        // Depending on requirements, you might implement retry logic here or log for manual follow-up
    }
}

// --- Utility Helper Functions ---

// Finds a value in an array of {key: ..., value: ...} objects (used for summaryRecordsMap)
function getValueByKey(recordsArray, searchKey) {
    if (!recordsArray || !Array.isArray(recordsArray)) return null;
    // Case-insensitive comparison might be safer depending on how keys are generated
    const record = recordsArray.find(item => item && typeof item.key === 'string' && item.key.toLowerCase() === searchKey.toLowerCase());
    return record ? record.value : null; // Return the value or null if not found
}

// Transforms the AI's quarterly output structure (for ONE quarter's AI result) into the format needed for Salesforce saving
function transformQuarterlyStructure(quarterlyAiOutput) {
    const result = {}; // Target structure: { year: { QX: { summaryDetails, summaryJson, count, startdate } } }

    // Add more robust validation based on your quarterlyFuncSchema structure
    // Ensure the expected nested structure exists before trying to access properties
    if (!quarterlyAiOutput?.yearlySummary?.[0]?.year ||
        !quarterlyAiOutput.yearlySummary[0]?.quarters?.[0]?.quarter ||
        !quarterlyAiOutput.yearlySummary[0]?.quarters?.[0]?.summary || // Check summary existence
        quarterlyAiOutput.yearlySummary[0]?.quarters?.[0]?.activityCount === undefined || // Check count existence (can be 0)
        !quarterlyAiOutput.yearlySummary[0]?.quarters?.[0]?.startdate // Check startdate existence
       ) {
        console.warn("[Transform Quarterly] Invalid or incomplete structure received from quarterly AI for transformation:", JSON.stringify(quarterlyAiOutput).substring(0, 500)); // Log truncated output
        return result; // Return empty object if essential parts are missing
    }

    try {
        // Access nested data safely now that we've validated the path
        const yearData = quarterlyAiOutput.yearlySummary[0];
        const year = yearData.year;
        result[year] = {}; // Initialize year object in the result

        const quarterData = yearData.quarters[0];
        const quarter = quarterData.quarter;

        // Extract the required fields
        const htmlSummary = quarterData.summary;
        // Stringify the *entire original AI output* for the Summary__c field to retain full context
        const fullQuarterlyJson = JSON.stringify(quarterlyAiOutput);
        const activityCount = quarterData.activityCount;
        const startDate = quarterData.startdate; // Expecting YYYY-MM-DD format from AI

        // Structure the data matching the keys used in createTimileSummarySalesforceRecords
        result[year][quarter] = {
            summaryDetails: htmlSummary,    // Extracted HTML summary for display field
            summaryJson: fullQuarterlyJson, // Full original AI JSON for storage/debug field
            count: activityCount,           // Aggregated activity count for the quarter
            startdate: startDate            // Start date of the quarter
        };

    } catch (transformError) {
        console.error("[Transform Quarterly] Error during quarterly AI output transformation:", transformError.stack);
        console.error("[Transform Quarterly] AI Output causing error (truncated):", JSON.stringify(quarterlyAiOutput).substring(0, 500));
        return {}; // Return empty object on unexpected error during transformation
    }

    // Return the structured data, e.g., { 2023: { Q1: { ...data... } } }
    return result;
}
