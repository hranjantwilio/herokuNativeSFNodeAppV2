/*
 * Enhanced Node.js Express application for generating Salesforce activity summaries using OpenAI Assistants.
 * Optimized for memory usage on platforms like Heroku.
 *
 * Features:
 * - Per-task OpenAI Assistant creation/deletion.
 * - Asynchronous processing with immediate acknowledgement.
 * - Salesforce integration (fetching activities, saving summaries).
 * - OpenAI Assistants API V2 usage.
 * - Dynamic function schema loading (default or from request).
 * - Dynamic tool_choice to force specific function calls (monthly/quarterly).
 * - Conditional input method: Direct JSON in prompt (< threshold) or File Upload (>= threshold).
 * - Incremental processing: Generates & saves monthly summaries individually, then aggregates & saves quarterly summaries individually.
 * - Robust error handling and callback mechanism.
 * - Temporary file management.
 */

// --- Dependencies ---
const express = require('express');
const jsforce = require('jsforce');
const dotenv = require('dotenv');
const { OpenAI, NotFoundError } = require("openai");
const fs = require("fs-extra");
const path = require("path");
const axios = require("axios");

// --- Load Environment Variables ---
dotenv.config();

// --- Configuration Constants ---
const SF_LOGIN_URL = process.env.SF_LOGIN_URL;
const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";
const TIMELINE_SUMMARY_OBJECT_API_NAME = "Timeline_Summary__c"; // Salesforce object API name
const DIRECT_INPUT_THRESHOLD = 2000; // Max activities for direct JSON input in prompt
const PROMPT_LENGTH_THRESHOLD = 256000; // Max characters for prompt before switching to file upload
const TEMP_FILE_DIR = path.join(__dirname, 'temp_files'); // Directory for temporary files

// --- Environment Variable Validation ---
if (!SF_LOGIN_URL || !OPENAI_API_KEY) {
    console.error("FATAL ERROR: Missing required environment variables (SF_LOGIN_URL, OPENAI_API_KEY).");
    process.exit(1);
}

// --- Default OpenAI Function Schemas ---
// (Keep your existing defaultFunctions array here)
const defaultFunctions = [
    {
      "name": "generate_monthly_activity_summary",
      "description": "Generates a structured monthly sales activity summary with insights and categorization based on provided activity data. Apply sub-theme segmentation within activityMapping.",
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
                          "LinkText": { "type": "string", "description": "'MMM DD YYYY: Short Description (max 50 chars)' - Generate from ActivityDate, Subject, Description." },
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
                            "LinkText": { "type": "string", "description": "'MMM DD YYYY: Short Description (max 50 chars)' - Generate from ActivityDate, Subject, Description." },
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
                            "LinkText": { "type": "string", "description": "'MMM DD YYYY: Short Description (max 50 chars)' - Generate from ActivityDate, Subject, Description." },
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
            "description": "Total number of activities processed for the month (matching the input count)."
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
app.use(express.json({ limit: '10mb' })); // Keep limit reasonable
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// --- Server Startup ---
app.listen(PORT, async () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Using OpenAI Model: ${OPENAI_MODEL}`);
    console.log(`Direct JSON input threshold: ${DIRECT_INPUT_THRESHOLD} activities`);
    console.log(`Prompt length threshold: ${PROMPT_LENGTH_THRESHOLD} characters`);
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
    console.log("Received /generatesummary request");

    // --- Authorization ---
    const authHeader = req.headers["authorization"];
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
        console.warn("Unauthorized request: Missing or invalid Bearer token.");
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
        summaryMap, // Optional JSON string map of existing summary records (e.g., {"Jan 2024": "recordId"})
        loggedinUserId,
        qtrJSON, // Optional override for quarterly function schema (JSON string)
        monthJSON // Optional override for monthly function schema (JSON string)
    } = req.body;

    if (!accountId || !callbackUrl || !accessToken || !queryText || !userPrompt || !userPromptQtr || !loggedinUserId) {
        console.warn("Bad Request: Missing required parameters.");
        return res.status(400).send({ error: "Missing required parameters (accountId, callbackUrl, accessToken, queryText, userPrompt, userPromptQtr, loggedinUserId)" });
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
            console.log("Using custom monthly function schema from request.");
        }
         if (qtrJSON) {
            quarterlyFuncSchema = JSON.parse(qtrJSON);
            if (!quarterlyFuncSchema || quarterlyFuncSchema.name !== 'generate_quarterly_activity_summary') {
                 throw new Error("Provided qtrJSON schema is invalid or missing the correct name property.");
            }
            console.log("Using custom quarterly function schema from request.");
        }
    } catch (e) {
        console.error("Failed to parse JSON input from request body:", e);
        return res.status(400).send({ error: `Invalid JSON provided in summaryMap, monthJSON, or qtrJSON. ${e.message}` });
    }

    // --- Ensure Schemas are Available ---
    if (!monthlyFuncSchema || !quarterlyFuncSchema) {
        console.error("FATAL: Default function schemas could not be loaded or found.");
        return res.status(500).send({ error: "Internal server error: Could not load function schemas."});
    }

    // --- Acknowledge Request (202 Accepted) ---
    res.status(202).json({ status: 'processing', message: 'Summary generation initiated. You will receive a callback.' });
    console.log(`Initiating summary processing for Account ID: ${accountId}`);

    // --- Start Asynchronous Processing ---
    processSummary(
        accountId,
        accessToken,
        callbackUrl,
        userPrompt,
        userPromptQtr,
        queryText,
        summaryRecordsMap,
        loggedinUserId,
        monthlyFuncSchema,
        quarterlyFuncSchema
    ).catch(async (error) => {
        console.error(`[${accountId}] Unhandled error during background processing:`, error);
        try {
            await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Unhandled processing error: ${error.message}`);
        } catch (callbackError) {
            console.error(`[${accountId}] Failed to send error callback after unhandled exception:`, callbackError);
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

// --- Asynchronous Summary Processing Logic (Refactored for Memory) ---
async function processSummary(
    accountId,
    accessToken,
    callbackUrl,
    userPromptMonthlyTemplate,
    userPromptQuarterlyTemplate,
    queryText,
    summaryRecordsMap,
    loggedinUserId,
    monthlyFuncSchema,
    quarterlyFuncSchema
) {
    console.log(`[${accountId}] Starting processSummary (Memory Optimized)...`);
    const conn = new jsforce.Connection({
        instanceUrl: SF_LOGIN_URL,
        accessToken: accessToken,
        maxRequest: 10, // Example: retry requests up to 10 times
    });

    // Store ONLY the AI JSON outputs needed for quarterly aggregation, grouped by quarter
    const quarterlyInputs = {}; // Structure: { "YYYY-QX": { monthSummaries: [aiOutput1, aiOutput2,...], year: Y, quarter: QX } }
    const monthMap = { january: 0, february: 1, march: 2, april: 3, may: 4, june: 5, july: 6, august: 7, september: 8, october: 9, november: 10, december: 11 };
    let overallStatus = "Success"; // Assume success initially
    let failureMessages = [];

    try {
        // 1. Fetch and Group
        console.log(`[${accountId}] Fetching Salesforce records...`);
        // Ensure groupRecordsByMonthYear minimizes data stored per activity
        const groupedData = await fetchRecords(conn, queryText);
        console.log(`[${accountId}] Fetched and grouped data by year/month.`);
        // Optional: Log memory usage after fetching/grouping
        // if (global.gc) { global.gc(); } // Force GC if needed (requires --expose-gc flag)
        // console.log(`[${accountId}] Memory usage after fetch/group:`, process.memoryUsage());


        // 2. Process Months Incrementally
        for (const year in groupedData) {
            console.log(`[${accountId}] Processing Year: ${year} for Monthly Summaries`);
            for (const monthObj of groupedData[year]) {
                for (const month in monthObj) {
                    let activities = monthObj[month]; // Activities for THIS month
                    const activityCount = activities.length;
                    console.log(`[${accountId}]   Processing Month: ${month} ${year} (${activityCount} activities)`);
                    if (activityCount === 0) continue;

                    const monthIndex = monthMap[month.toLowerCase()];
                    if (monthIndex === undefined) {
                        console.warn(`[${accountId}]   Could not map month name: ${month}. Skipping.`);
                        continue;
                    }
                    const startDate = new Date(Date.UTC(year, monthIndex, 1));
                    const startDateStr = startDate.toISOString().split('T')[0];
                    const userPromptMonthly = userPromptMonthlyTemplate.replace('{{YearMonth}}', `${month} ${year}`);

                    let monthlyAiOutput = null;
                    let monthlyAssistant = null;

                    try {
                        // *** Create Monthly Assistant ***
                        const assistantName = `Monthly Summarizer ${accountId} ${year}-${month}`;
                        console.log(`[${accountId}]   Creating Assistant: ${assistantName}`);
                        monthlyAssistant = await openai.beta.assistants.create({
                           name: assistantName,
                           instructions: "You are an AI assistant specialized in analyzing raw Salesforce activity data for a single month and generating structured JSON summaries using the provided function 'generate_monthly_activity_summary'. Apply sub-theme segmentation within the activityMapping as described in the function schema. Focus on extracting key themes, tone, and recommended actions.",
                           tools: [{ type: "file_search" }, { type: "function", "function": monthlyFuncSchema }],
                           model: OPENAI_MODEL,
                        });
                        console.log(`[${accountId}]   Created Assistant ID: ${monthlyAssistant.id}`);

                        // *** Generate Summary ***
                        monthlyAiOutput = await generateSummary(
                           activities,
                           openai,
                           monthlyAssistant.id,
                           userPromptMonthly,
                           monthlyFuncSchema
                       );
                       console.log(`[${accountId}]   Generated monthly summary for ${month} ${year}.`);

                        // *** Prepare for Salesforce Save (Single Month) ***
                        const monthlyForSalesforce = {
                           [year]: {
                               [month]: {
                                   summary: JSON.stringify(monthlyAiOutput), // Full JSON
                                   summaryDetails: monthlyAiOutput?.summary || '', // HTML part
                                   count: activityCount, // Use original count
                                   startdate: startDateStr
                               }
                           }
                        };

                        // *** Save THIS month's summary immediately ***
                        console.log(`[${accountId}]   Saving monthly summary for ${month} ${year} to Salesforce...`);
                        await createTimileSummarySalesforceRecords(conn, monthlyForSalesforce, accountId, 'Monthly', summaryRecordsMap);
                        console.log(`[${accountId}]   Saved monthly summary for ${month} ${year}.`);

                        // *** Store AI output for Quarterly Aggregation ***
                        const quarter = getQuarterFromMonthIndex(monthIndex);
                        const quarterKey = `${year}-${quarter}`;
                        if (!quarterlyInputs[quarterKey]) {
                            quarterlyInputs[quarterKey] = { monthSummaries: [], year: parseInt(year), quarter: quarter };
                        }
                        // Store the AI output needed by the quarterly function
                        quarterlyInputs[quarterKey].monthSummaries.push(monthlyAiOutput);

                    } catch (monthError) {
                        console.error(`[${accountId}] Error processing month ${month} ${year}:`, monthError);
                        overallStatus = "Failed"; // Mark overall process as failed
                        failureMessages.push(`Failed processing ${month} ${year}: ${monthError.message}`);
                        // Log and continue to the next month
                    } finally {
                        // *** Cleanup Monthly Assistant ***
                        if (monthlyAssistant?.id) {
                            try {
                                console.log(`[${accountId}]   Deleting monthly assistant ${monthlyAssistant.id}`);
                                await openai.beta.assistants.del(monthlyAssistant.id);
                                console.log(`[${accountId}]   Deleted monthly assistant ${monthlyAssistant.id}`);
                            } catch (delError) {
                                // Log deletion error but don't fail the process for this
                                console.warn(`[${accountId}]   Could not delete monthly assistant ${monthlyAssistant.id}:`, delError.message || delError);
                            }
                        }
                        // *** Explicitly release memory references ***
                        activities = null; // Allow GC to collect the activities array for this month
                        monthlyAiOutput = null; // Release reference to the AI output
                        // Optional: Trigger GC more aggressively if needed
                        // if (global.gc) { console.log(`[${accountId}] Triggering GC after month ${month}`); global.gc(); }
                        // console.log(`[${accountId}] Memory usage after month ${month}:`, process.memoryUsage());
                   }
                } // End month loop
            } // End monthObj loop
        } // End year loop

        // *** Release groupedData memory reference ***
        groupedData = null;
        // if (global.gc) { global.gc(); }
        // console.log(`[${accountId}] Memory usage after all months processed:`, process.memoryUsage());


        // 3. Process Quarters Incrementally
        console.log(`[${accountId}] Processing ${Object.keys(quarterlyInputs).length} quarters...`);
        for (const quarterKey in quarterlyInputs) {
            // Destructure needed data, including monthSummaries
            let { monthSummaries, year, quarter } = quarterlyInputs[quarterKey];
            const numMonthlySummaries = monthSummaries?.length || 0;
            console.log(`[${accountId}] Generating quarterly summary for ${quarterKey} using ${numMonthlySummaries} monthly summaries...`);

            if (numMonthlySummaries === 0) {
                console.warn(`[${accountId}] Skipping ${quarterKey} as it has no associated monthly summaries.`);
                continue;
            }

            // Prepare prompt including JSON data for the quarter
            const quarterlyInputDataString = JSON.stringify(monthSummaries, null, 2); // Stringify monthly outputs for prompt
            const userPromptQuarterly = `${userPromptQuarterlyTemplate.replace('{{Quarter}}', quarter).replace('{{Year}}', year)}\n\nAggregate the following monthly summary data provided below for ${quarterKey}:\n\`\`\`json\n${quarterlyInputDataString}\n\`\`\``;

            let quarterlyAiOutput = null;
            let quarterlyAssistant = null;

            try {
                // *** Create Quarterly Assistant ***
                const assistantName = `Quarterly Summarizer ${accountId} ${quarterKey}`;
                console.log(`[${accountId}]   Creating Assistant: ${assistantName}`);
                quarterlyAssistant = await openai.beta.assistants.create({
                    name: assistantName,
                    instructions: "You are an AI assistant specialized in aggregating pre-summarized monthly Salesforce activity data (provided as JSON in the prompt) into a structured quarterly JSON summary for a specific quarter using the provided function 'generate_quarterly_activity_summary'. Consolidate insights and activity lists accurately based on the input monthly summaries.",
                    tools: [{ type: "file_search" }, { type: "function", "function": quarterlyFuncSchema }],
                    model: OPENAI_MODEL,
                });
                console.log(`[${accountId}]   Created Assistant ID: ${quarterlyAssistant.id}`);

                // *** Generate Summary ***
                quarterlyAiOutput = await generateSummary(
                   null, // No raw activities needed, data is in prompt
                   openai,
                   quarterlyAssistant.id,
                   userPromptQuarterly,
                   quarterlyFuncSchema
                );
                console.log(`[${accountId}] Successfully generated quarterly summary AI output for ${quarterKey}.`);

                // *** Transform and Prepare for Save (Single Quarter) ***
                const transformedQuarter = transformQuarterlyStructure(quarterlyAiOutput);

                // Check if transform produced valid output for THIS quarter
                if (transformedQuarter && transformedQuarter[year] && transformedQuarter[year][quarter]) {
                    const quarterlyForSalesforce = {
                        [year]: {
                            [quarter]: transformedQuarter[year][quarter]
                        }
                    };
                    // *** Save THIS quarter's summary immediately ***
                    console.log(`[${accountId}] Saving quarterly summary for ${quarterKey} to Salesforce...`);
                    await createTimileSummarySalesforceRecords(conn, quarterlyForSalesforce, accountId, 'Quarterly', summaryRecordsMap);
                    console.log(`[${accountId}] Saved quarterly summary for ${quarterKey}.`);
                } else {
                     console.warn(`[${accountId}] Quarterly summary for ${quarterKey} generated by AI but transform failed or produced empty/invalid structure. Skipping save.`);
                     // Optionally mark as partial failure
                     if (overallStatus !== "Failed") { overallStatus = "Partial Success"; }
                     failureMessages.push(`Failed to transform/validate AI output for ${quarterKey}.`);
                }

            } catch (quarterlyError) {
                console.error(`[${accountId}] Failed to generate or save quarterly summary for ${quarterKey}:`, quarterlyError);
                overallStatus = "Failed"; // Mark overall process as failed
                failureMessages.push(`Failed processing ${quarterKey}: ${quarterlyError.message}`);
                // Log and continue to next quarter
            } finally {
                // *** Cleanup Quarterly Assistant ***
                 if (quarterlyAssistant?.id) {
                    try {
                        console.log(`[${accountId}]   Deleting quarterly assistant ${quarterlyAssistant.id}`);
                        await openai.beta.assistants.del(quarterlyAssistant.id);
                        console.log(`[${accountId}]   Deleted quarterly assistant ${quarterlyAssistant.id}`);
                    } catch (delError) {
                        console.warn(`[${accountId}]   Could not delete quarterly assistant ${quarterlyAssistant.id}:`, delError.message || delError);
                    }
                 }
                 // *** Release memory references for this quarter ***
                 // Clear the array holding the monthly summaries for this quarter
                 if (quarterlyInputs[quarterKey]) {
                     quarterlyInputs[quarterKey].monthSummaries = null;
                 }
                 quarterlyAiOutput = null; // Release reference to the AI output
                 monthSummaries = null; // Release local reference
                 // Optional: Trigger GC
                 // if (global.gc) { console.log(`[${accountId}] Triggering GC after quarter ${quarterKey}`); global.gc(); }
                 // console.log(`[${accountId}] Memory usage after quarter ${quarterKey}:`, process.memoryUsage());
           }
        } // End quarter loop

        // 4. Send Final Callback
        console.log(`[${accountId}] Process completed.`);
        let finalMessage = overallStatus === "Success" ? "Summary Processed Successfully" : `Processing finished with issues: ${failureMessages.join('; ')}`;
        if (finalMessage.length > 1000) { // Avoid overly long callback messages
            finalMessage = finalMessage.substring(0, 997) + "...";
        }
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, overallStatus, finalMessage);

    } catch (error) {
        // Catch errors from initial fetch/group or other unhandled exceptions
        console.error(`[${accountId}] Critical error during summary processing:`, error);
        // Ensure a failure callback is sent
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Critical processing error: ${error.message}`);
    } finally {
         console.log(`[${accountId}] processSummary finished execution.`);
         // Optional: Force GC at the very end if needed
         // if (global.gc) { console.log(`[${accountId}] Triggering final GC.`); global.gc(); }
         // console.log(`[${accountId}] Final memory usage:`, process.memoryUsage());
    }
}


// --- OpenAI Summary Generation Function (Includes conditional input) ---
async function generateSummary(
    activities, // Array of raw activities OR null
    openaiClient,
    assistantId,
    userPrompt,
    functionSchema
) {
    let fileId = null;
    let thread = null;
    let tempFilePath = null; // Use a distinct name
    let inputMethod = "prompt";
    // Use the configured threshold constants
    // const PROMPT_LENGTH_THRESHOLD = 256000; declared globally
    // const DIRECT_INPUT_THRESHOLD = 2000; declared globally

    try {
        thread = await openaiClient.beta.threads.create();
        console.log(`[Thread ${thread.id}] Created for Assistant ${assistantId}`);

        let finalUserPrompt = userPrompt;
        let messageAttachments = [];

        if (activities && Array.isArray(activities) && activities.length > 0) {
            let potentialFullPrompt;
            let activitiesJsonString;

            try {
                 activitiesJsonString = JSON.stringify(activities); // Stringify without pretty print first for length check
                 potentialFullPrompt = `${userPrompt}\n\nHere is the activity data to process:\n\`\`\`json\n${activitiesJsonString}\n\`\`\``;
                 console.log(`[Thread ${thread.id}] Potential prompt length with direct JSON: ${potentialFullPrompt.length} characters.`);
            } catch(stringifyError) {
                console.error(`[Thread ${thread.id}] Error stringifying activities for length check:`, stringifyError);
                throw new Error("Failed to stringify activity data for processing.");
            }

            if (potentialFullPrompt.length < PROMPT_LENGTH_THRESHOLD && activities.length < DIRECT_INPUT_THRESHOLD) {
                inputMethod = "direct JSON";
                // Use the prompt with embedded JSON (maybe pretty print now for AI readability if desired)
                finalUserPrompt = `${userPrompt}\n\nHere is the activity data to process:\n\`\`\`json\n${JSON.stringify(activities, null, 2)}\n\`\`\``;
                console.log(`[Thread ${thread.id}] Using direct JSON input.`);
            } else {
                inputMethod = "file upload";
                console.log(`[Thread ${thread.id}] Using file upload (Activities: ${activities.length} >= ${DIRECT_INPUT_THRESHOLD} or Prompt Length: ${potentialFullPrompt.length} >= ${PROMPT_LENGTH_THRESHOLD}).`);
                // Use the *original* base userPrompt
                finalUserPrompt = userPrompt;

                // Convert activities to Plain Text for file_search
                let activitiesText = activities.map((activity, index) => {
                    let activityLines = [`Activity ${index + 1} (ID: ${activity.Id || 'N/A'}):`];
                    // Only include fields relevant to the AI's task
                    activityLines.push(`  ActivityDate: ${activity.ActivityDate || 'N/A'}`);
                    activityLines.push(`  Subject: ${activity.Subject || 'No Subject'}`);
                    activityLines.push(`  Description: ${activity.Description || 'No Description'}`);
                    // Add other relevant fields here if needed by the AI
                    return activityLines.join('\n');
                }).join('\n\n---\n\n'); // Separator between activities

                const timestamp = new Date().toISOString().replace(/[:.-]/g, "_");
                // Use .txt extension for file_search
                const filename = `activities_${accountId}_${timestamp}_${thread.id}.txt`; // More specific filename
                tempFilePath = path.join(TEMP_FILE_DIR, filename); // Use configured temp dir

                await fs.writeFile(tempFilePath, activitiesText);
                console.log(`[Thread ${thread.id}] Temporary text file generated: ${tempFilePath}`);

                const fileStream = fs.createReadStream(tempFilePath);
                const uploadResponse = await openaiClient.files.create({
                    file: fileStream,
                    purpose: "assistants", // Correct purpose
                });
                fileId = uploadResponse.id;
                console.log(`[Thread ${thread.id}] File uploaded to OpenAI: ${fileId}`);

                // Attach file using file_search tool
                messageAttachments.push({ file_id: fileId, tools: [{ type: "file_search" }] });
                console.log(`[Thread ${thread.id}] Attaching file ${fileId} with file_search tool.`);

                // Instruct AI to use the file if needed
                finalUserPrompt += "\n\nPlease analyze the activity data provided in the attached file.";
            }
        } else {
             console.log(`[Thread ${thread.id}] No activities array provided or array is empty. Using prompt content as is.`);
        }

        const messagePayload = { role: "user", content: finalUserPrompt };
        if (messageAttachments.length > 0) {
            messagePayload.attachments = messageAttachments;
        }

        const message = await openaiClient.beta.threads.messages.create(thread.id, messagePayload);
        console.log(`[Thread ${thread.id}] Message added (using ${inputMethod}). ID: ${message.id}`);

        console.log(`[Thread ${thread.id}] Starting run, forcing function: ${functionSchema.name}`);
        const run = await openaiClient.beta.threads.runs.createAndPoll(thread.id, {
            assistant_id: assistantId,
            // Provide the specific function schema for this run
            tools: [{ type: "function", function: functionSchema }],
            // Force the specific function call
            tool_choice: { type: "function", function: { name: functionSchema.name } },
        });
        console.log(`[Thread ${thread.id}] Run status: ${run.status}`);

        if (run.status === 'requires_action') {
            const toolCalls = run.required_action?.submit_tool_outputs?.tool_calls;
            if (!toolCalls || toolCalls.length === 0 || !toolCalls[0]?.function?.arguments) {
                 console.error(`[Thread ${thread.id}] Run requires action, but tool call data is missing or invalid.`, run.required_action);
                 throw new Error("Function call was expected but not provided correctly by the Assistant.");
             }
             const toolCall = toolCalls[0];
             if (toolCall.function.name !== functionSchema.name) {
                  console.error(`[Thread ${thread.id}] Assistant called the wrong function. Expected: ${functionSchema.name}, Got: ${toolCall.function.name}`);
                  throw new Error(`Assistant called the wrong function: ${toolCall.function.name}`);
             }
             const rawArgs = toolCall.function.arguments;
             console.log(`[Thread ${thread.id}] Function call arguments received for ${toolCall.function.name}. Length: ${rawArgs.length}`);
             try {
                 const summaryObj = JSON.parse(rawArgs);
                 console.log(`[Thread ${thread.id}] Successfully parsed function arguments.`);
                 return summaryObj;
             } catch (parseError) {
                 console.error(`[Thread ${thread.id}] Failed to parse function call arguments JSON:`, parseError);
                 console.error(`[Thread ${thread.id}] Raw arguments received (first 500 chars):`, rawArgs.substring(0, 500));
                 throw new Error(`Failed to parse function call arguments from AI: ${parseError.message}`);
             }
         } else if (run.status === 'completed') {
              console.warn(`[Thread ${thread.id}] Run completed without requiring function call action, despite tool_choice forcing ${functionSchema.name}.`);
              // Attempt to get the last message from the assistant for debugging
              try {
                const messages = await openaiClient.beta.threads.messages.list(run.thread_id, { order: 'desc', limit: 1 });
                const lastMessageContent = messages.data[0]?.content[0]?.text?.value || "No text content found.";
                console.warn(`[Thread ${thread.id}] Last message content from Assistant: ${lastMessageContent.substring(0, 500)}...`);
              } catch (msgError) {
                  console.warn(`[Thread ${thread.id}] Could not retrieve last message for completed run: ${msgError.message}`);
              }
              throw new Error(`Assistant run completed without making the required function call to ${functionSchema.name}. Check Assistant response or instructions.`);
         } else {
             console.error(`[Thread ${thread.id}] Run failed or ended unexpectedly. Status: ${run.status}`, run.last_error || run.incomplete_details);
             const errorMessage = run.last_error ? `${run.last_error.code}: ${run.last_error.message}` : (run.incomplete_details?.reason || 'Unknown error');
             throw new Error(`Assistant run failed. Status: ${run.status}. Error: ${errorMessage}`);
         }

    } catch (error) {
        console.error(`[Thread ${thread?.id || 'N/A'}] Error in generateSummary: ${error.message}`);
        // Log stack trace for better debugging
        console.error(error.stack);
        throw error; // Re-throw
    } finally {
        // Cleanup temporary local file
        if (tempFilePath) {
            try {
                await fs.unlink(tempFilePath);
                console.log(`[Thread ${thread?.id || 'N/A'}] Deleted temporary file: ${tempFilePath}`);
            } catch (unlinkError) {
                console.error(`[Thread ${thread?.id || 'N/A'}] Error deleting temporary file ${tempFilePath}:`, unlinkError);
            }
        }
        // Cleanup OpenAI file
        if (fileId) {
            try {
                await openaiClient.files.del(fileId);
                console.log(`[Thread ${thread?.id || 'N/A'}] Deleted OpenAI file: ${fileId}`);
            } catch (deleteError) {
                 if (!(deleteError instanceof NotFoundError || deleteError?.status === 404)) {
                    console.error(`[Thread ${thread?.id || 'N/A'}] Error deleting OpenAI file ${fileId}:`, deleteError.message || deleteError);
                 } else {
                     console.log(`[Thread ${thread?.id || 'N/A'}] OpenAI file ${fileId} already deleted or not found.`);
                 }
            }
        }
        // Optional: Delete the thread itself if resource management is critical
        // if (thread?.id) { try { await openaiClient.beta.threads.del(thread.id); console.log(`[Thread ${thread.id}] Deleted thread.`); } catch(e){ console.warn(`Failed to delete thread ${thread.id}`)} }
    }
}


// --- Salesforce Record Creation/Update Function (Bulk API) ---
// (Keep your existing createTimileSummarySalesforceRecords function - it should work well with incremental saves)
async function createTimileSummarySalesforceRecords(conn, summaries, parentId, summaryCategory, summaryRecordsMap) {
    console.log(`[${parentId}] Preparing to save ${summaryCategory} summaries...`);
    let recordsToCreate = [];
    let recordsToUpdate = [];

    // Iterate through the summaries structure { year: { periodKey: { summaryJson/summary, summaryDetails, count, startdate } } }
    for (const year in summaries) {
        for (const periodKey in summaries[year]) { // periodKey is 'MonthName' or 'Q1', 'Q2' etc.
            const summaryData = summaries[year][periodKey];

            let summaryJsonString = summaryData.summaryJson || summaryData.summary; // Full AI response JSON
            let summaryDetailsHtml = summaryData.summaryDetails || ''; // Extracted HTML summary
            let startDate = summaryData.startdate; // Should be YYYY-MM-DD
            let count = summaryData.count;

             // Fallback: Try to extract HTML from the full JSON if details field is empty
             if (!summaryDetailsHtml && summaryJsonString) {
                try {
                    const parsedJson = JSON.parse(summaryJsonString);
                    summaryDetailsHtml = parsedJson.summary || ''; // Assumes top-level 'summary' key holds HTML
                    // For quarterly, the path might be different based on transformQuarterlyStructure
                    if (summaryCategory === 'Quarterly' && !summaryDetailsHtml && parsedJson?.yearlySummary?.[0]?.quarters?.[0]?.summary) {
                        summaryDetailsHtml = parsedJson.yearlySummary[0].quarters[0].summary;
                    }
                } catch (e) {
                    console.warn(`[${parentId}] Could not parse 'summaryJsonString' for ${periodKey} ${year} to extract HTML details. HTML field might be empty.`);
                 }
            }

            let fyQuarterValue = (summaryCategory === 'Quarterly') ? periodKey : '';
            let monthValue = (summaryCategory === 'Monthly') ? periodKey : '';
            let shortMonth = monthValue ? monthValue.substring(0, 3) : '';
            let summaryMapKey = (summaryCategory === 'Quarterly') ? `${fyQuarterValue} ${year}` : `${shortMonth} ${year}`;
            let existingRecordId = getValueByKey(summaryRecordsMap, summaryMapKey);

            // Prepare Salesforce Record Payload (Ensure API names match your org)
            const recordPayload = {
                Parent_Id__c: parentId,
                Month__c: monthValue || null, // Use null if empty
                Year__c: String(year),
                Summary_Category__c: summaryCategory,
                // Use substring to prevent exceeding Salesforce field limits
                Summary__c: summaryJsonString ? summaryJsonString.substring(0, 131070) : null,
                Summary_Details__c: summaryDetailsHtml ? summaryDetailsHtml.substring(0, 131070) : null,
                FY_Quarter__c: fyQuarterValue || null, // Use null if empty
                Month_Date__c: startDate,
                Number_of_Records__c: count,
                Account__c: parentId // Adjust if relationship field name is different
            };

             if (!recordPayload.Parent_Id__c || !recordPayload.Summary_Category__c || !recordPayload.Year__c) {
                 console.warn(`[${parentId}] Skipping record for ${summaryMapKey} due to missing Parent ID, Category, or Year.`);
                 continue;
             }

            if (existingRecordId) {
                console.log(`[${parentId}]   Queueing update for ${summaryMapKey} (ID: ${existingRecordId})`);
                recordsToUpdate.push({ Id: existingRecordId, ...recordPayload });
            } else {
                console.log(`[${parentId}]   Queueing create for ${summaryMapKey}`);
                recordsToCreate.push(recordPayload);
            }
        }
    }

    // Perform Bulk DML Operations
    try {
        const options = { allOrNone: false }; // Allow partial success

        if (recordsToCreate.length > 0) {
            console.log(`[${parentId}] Creating ${recordsToCreate.length} new ${summaryCategory} summary records via bulk API...`);
            const createResults = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "insert", options, recordsToCreate);
            console.log(`[${parentId}] Bulk create results received (${createResults.length}).`);
            createResults.forEach((res, index) => {
                if (!res.success) {
                    const recordIdentifier = recordsToCreate[index].Month__c || recordsToCreate[index].FY_Quarter__c;
                    console.error(`[${parentId}] Error creating record ${index + 1} (${recordIdentifier} ${recordsToCreate[index].Year__c}):`, res.errors);
                    // Optionally add to failure messages in the main process
                }
            });
        } else {
            console.log(`[${parentId}] No new ${summaryCategory} records to create.`);
        }

        if (recordsToUpdate.length > 0) {
            console.log(`[${parentId}] Updating ${recordsToUpdate.length} existing ${summaryCategory} summary records via bulk API...`);
             const updateResults = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "update", options, recordsToUpdate);
             console.log(`[${parentId}] Bulk update results received (${updateResults.length}).`);
             updateResults.forEach((res, index) => {
                 if (!res.success) {
                    console.error(`[${parentId}] Error updating record ${index + 1} (ID: ${recordsToUpdate[index].Id}):`, res.errors);
                    // Optionally add to failure messages
                 }
             });
        } else {
            console.log(`[${parentId}] No existing ${summaryCategory} records to update.`);
        }
    } catch (err) {
        console.error(`[${parentId}] Failed to save ${summaryCategory} records to Salesforce using Bulk API: ${err.message}`, err.stack);
        // Throw error to be caught by processSummary and trigger failure callback
        throw new Error(`Salesforce save operation failed: ${err.message}`);
    }
}

// --- Salesforce Data Fetching with Pagination ---
// (Keep existing fetchRecords - it fetches all needed data upfront)
async function fetchRecords(conn, queryOrUrl, allRecords = [], isFirstIteration = true) {
    try {
        const logPrefix = isFirstIteration ? `Initial query` : `Querying more records from nextRecordsUrl`;
        console.log(`[SF Fetch] ${logPrefix}...`);

        const queryResult = isFirstIteration
            ? await conn.query(queryOrUrl)
            : await conn.queryMore(queryOrUrl);

        const fetchedCount = queryResult.records ? queryResult.records.length : 0;
        if (fetchedCount > 0) {
            allRecords = allRecords.concat(queryResult.records);
        }
        console.log(`[SF Fetch] Fetched ${fetchedCount} records. Total so far: ${allRecords.length}. Done: ${queryResult.done}`);


        if (!queryResult.done && queryResult.nextRecordsUrl) {
            // Tail recursion optimization might not apply perfectly in JS async/await,
            // but passing the growing array avoids creating copies at each step.
            return fetchRecords(conn, queryResult.nextRecordsUrl, allRecords, false);
        } else {
            console.log(`[SF Fetch] Finished fetching. Total records retrieved: ${allRecords.length}. Grouping...`);
            // Group AFTER all records are fetched
            return groupRecordsByMonthYear(allRecords);
        }
    } catch (error) {
        console.error(`[SF Fetch] Error fetching Salesforce activities: ${error.message}`, error.stack);
        throw error;
    }
}


// --- Data Grouping Helper Function (Minimize stored data) ---
function groupRecordsByMonthYear(records) {
    const groupedData = {}; // { year: [ { MonthName: [activityObj, ...] }, ... ], ... }
    records.forEach(activity => {
        if (!activity.ActivityDate) {
            console.warn(`Skipping activity (ID: ${activity.Id || 'Unknown'}) due to missing ActivityDate.`);
            return;
        }
        try {
            const date = new Date(activity.ActivityDate);
             if (isNaN(date.getTime())) {
                 console.warn(`Skipping activity (ID: ${activity.Id || 'Unknown'}) due to invalid ActivityDate format: ${activity.ActivityDate}`);
                 return;
             }

            const year = date.getUTCFullYear();
            const month = date.toLocaleString('en-US', { month: 'long', timeZone: 'UTC' });

            if (!groupedData[year]) {
                groupedData[year] = [];
            }

            let monthEntry = groupedData[year].find(entry => entry[month]);
            if (!monthEntry) {
                monthEntry = { [month]: [] };
                groupedData[year].push(monthEntry);
            }

            // *** Crucial for Memory: Only store essential fields needed by the AI ***
            // Adjust based on your monthlyFuncSchema requirements
            monthEntry[month].push({
                Id: activity.Id,
                // Only include Description/Subject if necessary for the AI.
                // Consider truncating if full text isn't needed initially to save memory.
                Description: activity.Description || null, // Keep null if empty
                Subject: activity.Subject || null,     // Keep null if empty
                ActivityDate: activity.ActivityDate // YYYY-MM-DD format
                // Add other *essential* fields here
            });
        } catch(dateError) {
             console.warn(`Skipping activity (ID: ${activity.Id || 'Unknown'}) due to date processing error: ${dateError.message}. Date value: ${activity.ActivityDate}`);
        }
    });
    console.log("Finished grouping records by year and month.");
    return groupedData;
}


// --- Callback Sending Function ---
// (Keep existing sendCallbackResponse - looks okay)
async function sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, status, message) {
    const logMessage = message.length > 200 ? message.substring(0, 200) + '...' : message;
    console.log(`[${accountId}] Sending callback to ${callbackUrl}. Status: ${status}, Message: ${logMessage}`);
    try {
        await axios.post(callbackUrl,
            {
                accountId: accountId,
                loggedinUserId: loggedinUserId,
                status: "Completed", // Status of the callback action itself
                processResult: status, // Overall result ('Success', 'Failed', 'Partial Success')
                message: message
            },
            {
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${accessToken}` // Use SF token for callback auth
                },
                timeout: 30000 // Increased timeout for callback
            }
        );
        console.log(`[${accountId}] Callback sent successfully.`);
    } catch (error) {
        let errorMessage = error.message;
        if (error.response) {
            errorMessage = `Status: ${error.response.status}, Data: ${JSON.stringify(error.response.data)}, Message: ${error.message}`;
        } else if (error.request) {
            errorMessage = `No response received from callback URL. ${error.message}`;
        }
        console.error(`[${accountId}] Failed to send callback to ${callbackUrl}: ${errorMessage}`);
        // Consider retry logic or alternative notification for failed callbacks
    }
}


// --- Utility Helper Functions ---

// Finds a value in an array of {key: ..., value: ...} objects
function getValueByKey(recordsArray, searchKey) {
    if (!recordsArray || !Array.isArray(recordsArray)) return null;
    const record = recordsArray.find(item => item && item.key === searchKey);
    return record ? record.value : null;
}

// Transforms the AI's quarterly output structure (for ONE quarter's AI result)
function transformQuarterlyStructure(quarterlyAiOutput) {
    const result = {}; // { year: { QX: { summaryDetails, summaryJson, count, startdate } } }

    // Add more robust validation based on your quarterlyFuncSchema structure
    if (!quarterlyAiOutput?.yearlySummary?.[0]?.year || !quarterlyAiOutput.yearlySummary[0]?.quarters?.[0]?.quarter) {
        console.warn("Invalid or incomplete structure received from quarterly AI for transformation:", JSON.stringify(quarterlyAiOutput));
        return result; // Return empty if essential parts are missing
    }

    try {
        const yearData = quarterlyAiOutput.yearlySummary[0];
        const year = yearData.year;
        result[year] = {};

        const quarterData = yearData.quarters[0];
        const quarter = quarterData.quarter;

        if (!quarterData.summary || !quarterData.activityCount === undefined || !quarterData.startdate) {
             console.warn(`Invalid quarter data fields in quarterly AI output passed to transform for year ${year}, quarter ${quarter}:`, quarterData);
             // Return potentially with just the empty year object if quarter details are invalid
             return result;
        }

        const htmlSummary = quarterData.summary;
        // Stringify the *entire original AI output* for the Summary__c field for full context
        const fullQuarterlyJson = JSON.stringify(quarterlyAiOutput);
        const activityCount = quarterData.activityCount;
        const startDate = quarterData.startdate; // Expecting YYYY-MM-DD

        // Structure for createTimileSummarySalesforceRecords
        result[year][quarter] = {
            summaryDetails: htmlSummary,    // Extracted HTML
            summaryJson: fullQuarterlyJson, // Full original AI JSON
            count: activityCount,
            startdate: startDate
        };

    } catch (transformError) {
        console.error("Error during quarterly AI output transformation:", transformError);
        console.error("AI Output causing error:", JSON.stringify(quarterlyAiOutput));
        return {}; // Return empty on unexpected error
    }

    return result; // e.g., { 2023: { Q1: { ...data... } } }
}

// Helper to get start month (not used by transformQuarterlyStructure anymore as startdate is expected from AI)
/*
function getQuarterStartMonth(quarter) {
    // ... (keep if needed elsewhere, otherwise removable)
}
*/
