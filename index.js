/*
 * Enhanced Node.js Express application for generating Salesforce activity summaries using OpenAI Assistants.
 *
 * Features:
 * - Retrieves pre-configured OpenAI Assistant IDs from environment variables on startup.
 * - Asynchronous processing with immediate acknowledgement.
 * - Salesforce integration (fetching activities, saving summaries).
 * - OpenAI Assistants API V2 usage.
 * - Dynamic function schema loading (default or from request).
 * - Dynamic tool_choice to force specific function calls (monthly/quarterly).
 * - Conditional input method: Direct JSON in prompt (< threshold) or File Upload (>= threshold).
 * - Generates summaries per month and aggregates per relevant quarter individually.
 * - Robust error handling and callback mechanism.
 * - Temporary file management.
 */

// --- Dependencies ---
const express = require('express');
const jsforce = require('jsforce');
const dotenv = require('dotenv');
const { OpenAI, NotFoundError } = require("openai"); // Import NotFoundError specifically
const fs = require("fs-extra"); // Using fs-extra for promise-based file operations and JSON handling
const path = require("path");
const axios = require("axios");

// --- Load Environment Variables ---
dotenv.config();

// --- Configuration Constants ---
const SF_LOGIN_URL = process.env.SF_LOGIN_URL;
const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MONTHLY_ASSISTANT_ID = process.env.OPENAI_MONTHLY_ASSISTANT_ID; // <-- New
const OPENAI_QUARTERLY_ASSISTANT_ID = process.env.OPENAI_QUARTERLY_ASSISTANT_ID; // <-- New

const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o"; // Model used by the Assistants (ensure it matches your pre-configured ones)
const TIMELINE_SUMMARY_OBJECT_API_NAME = "Timeline_Summary__c"; // Salesforce object API name
const DIRECT_INPUT_THRESHOLD = 2000; // Max activities for direct JSON input in prompt
const PROMPT_LENGTH_THRESHOLD = 256000; // Character limit for direct prompt input
const TEMP_FILE_DIR = path.join(__dirname, 'temp_files'); // Directory for temporary files

// --- Environment Variable Validation ---
if (!SF_LOGIN_URL || !OPENAI_API_KEY || !OPENAI_MONTHLY_ASSISTANT_ID || !OPENAI_QUARTERLY_ASSISTANT_ID) {
    console.error("FATAL ERROR: Missing required environment variables (SF_LOGIN_URL, OPENAI_API_KEY, OPENAI_MONTHLY_ASSISTANT_ID, OPENAI_QUARTERLY_ASSISTANT_ID).");
    process.exit(1); // Exit if essential config is missing
}

// --- Global Variables for Assistant IDs ---
let monthlyAssistantId = OPENAI_MONTHLY_ASSISTANT_ID;
let quarterlyAssistantId = OPENAI_QUARTERLY_ASSISTANT_ID;


// --- Default OpenAI Function Schemas ---
// These define the structure the AI is *expected* to return via the function call.
// The actual Assistants (monthly/quarterly) must be configured in OpenAI to use
// the 'function' tool capability. The *specific* function schema is passed during the run.
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
app.use(express.json({ limit: '10mb' })); // Increase JSON payload limit
app.use(express.urlencoded({ extended: true, limit: '10mb' }));


// --- Helper to Verify Assistant Existence on Startup ---
async function verifyAssistantsExist() {
    let monthlyOk = false;
    let quarterlyOk = false;
    console.log("Verifying OpenAI Assistant IDs...");
    try {
        await openai.beta.assistants.retrieve(monthlyAssistantId);
        console.log(` - Monthly Assistant (${monthlyAssistantId}) found.`);
        monthlyOk = true;
    } catch (error) {
        if (error instanceof NotFoundError) {
            console.error(`FATAL ERROR: Monthly Assistant with ID "${monthlyAssistantId}" not found in OpenAI.`);
        } else {
            console.error(`FATAL ERROR: Error retrieving Monthly Assistant "${monthlyAssistantId}":`, error.message);
        }
    }
    try {
        await openai.beta.assistants.retrieve(quarterlyAssistantId);
        console.log(` - Quarterly Assistant (${quarterlyAssistantId}) found.`);
        quarterlyOk = true;
    } catch (error) {
        if (error instanceof NotFoundError) {
            console.error(`FATAL ERROR: Quarterly Assistant with ID "${quarterlyAssistantId}" not found in OpenAI.`);
        } else {
            console.error(`FATAL ERROR: Error retrieving Quarterly Assistant "${quarterlyAssistantId}":`, error.message);
        }
    }

    if (!monthlyOk || !quarterlyOk) {
        console.error("FATAL ERROR: One or more required OpenAI Assistants could not be verified. Ensure the IDs in environment variables are correct and the Assistants exist.");
        console.error("Please create the Assistants in OpenAI with appropriate instructions, model (e.g., gpt-4o), and enable 'Function calling' and 'File search' tools, then update the .env file.");
        process.exit(1); // Exit if verification fails
    }
    console.log("OpenAI Assistant verification successful.");
}


// --- Server Startup ---
// Wrap startup in an async IIFE to allow await for verification
(async () => {
    await verifyAssistantsExist(); // Verify assistants before starting the server

    app.listen(PORT, () => {
        console.log(`Server running on port ${PORT}`);
        console.log(`Using OpenAI Model (configured on Assistants): ${OPENAI_MODEL}`);
        console.log(`Using Monthly Assistant ID: ${monthlyAssistantId}`);
        console.log(`Using Quarterly Assistant ID: ${quarterlyAssistantId}`);
        console.log(`Direct JSON input threshold: ${DIRECT_INPUT_THRESHOLD} activities`);
        console.log(`Prompt length threshold for file upload: ${PROMPT_LENGTH_THRESHOLD} characters`);
    });
})();


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
            if (!monthlyFuncSchema || typeof monthlyFuncSchema !== 'object' || !monthlyFuncSchema.name) {
                throw new Error("Provided monthJSON schema is invalid or missing the 'name' property.");
            }
             // Optional: Deeper validation of the schema structure if needed
            console.log("Using custom monthly function schema from request.");
        }
         if (qtrJSON) {
            quarterlyFuncSchema = JSON.parse(qtrJSON);
             if (!quarterlyFuncSchema || typeof quarterlyFuncSchema !== 'object' || !quarterlyFuncSchema.name) {
                 throw new Error("Provided qtrJSON schema is invalid or missing the 'name' property.");
            }
             // Optional: Deeper validation
            console.log("Using custom quarterly function schema from request.");
        }
    } catch (e) {
        console.error("Failed to parse JSON input from request body:", e);
        return res.status(400).send({ error: `Invalid JSON provided in summaryMap, monthJSON, or qtrJSON. ${e.message}` });
    }

     // --- Ensure Schemas are Available (Default or Custom) ---
     if (!monthlyFuncSchema || !quarterlyFuncSchema) {
         // This should theoretically not happen if defaults are present and parsing is checked
         console.error("FATAL: Function schemas could not be loaded or parsed correctly.");
         return res.status(500).send({ error: "Internal server error: Could not load function schemas."});
     }

    // --- Acknowledge Request (202 Accepted) ---
    res.status(202).json({ status: 'processing', message: 'Summary generation initiated. You will receive a callback.' });
    console.log(`Initiating summary processing for Account ID: ${accountId}`);

    // --- Start Asynchronous Processing ---
    // Pass the validated, globally stored assistant IDs
    processSummary(
        accountId,
        accessToken,
        callbackUrl,
        userPrompt,
        userPromptQtr,
        queryText,
        summaryRecordsMap,
        loggedinUserId,
        monthlyFuncSchema, // Pass the potentially customized schema
        quarterlyFuncSchema, // Pass the potentially customized schema
        monthlyAssistantId, // Pass the verified ID
        quarterlyAssistantId // Pass the verified ID
    ).catch(async (error) => {
        // Catch unhandled errors from the top level of processSummary
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
    return 'Unknown'; // Should not happen with valid index
}

// --- Asynchronous Summary Processing Logic ---
async function processSummary(
    accountId,
    accessToken,
    callbackUrl,
    userPromptMonthlyTemplate,
    userPromptQuarterlyTemplate,
    queryText,
    summaryRecordsMap,
    loggedinUserId,
    monthlyFuncSchema, // Receive the final schema to use
    quarterlyFuncSchema, // Receive the final schema to use
    verifiedMonthlyAssistantId, // Receive the verified ID
    verifiedQuarterlyAssistantId // Receive the verified ID
) {
    console.log(`[${accountId}] Starting processSummary...`);
    // Use the verified Assistant IDs passed as arguments

    const conn = new jsforce.Connection({
        instanceUrl: SF_LOGIN_URL,
        accessToken: accessToken,
        maxRequest: 5, // Add some retry logic for Salesforce requests
        version: '59.0' // Specify API version
    });

    try {
        // 1. Fetch Salesforce Records
        console.log(`[${accountId}] Fetching Salesforce records...`);
        const groupedData = await fetchRecords(conn, queryText);
        console.log(`[${accountId}] Fetched and grouped data by year/month.`);

        // 2. Generate Monthly Summaries
        const finalMonthlySummaries = {}; // Structure: { year: { month: { aiOutput: {}, count: N, startdate: "...", year: Y, monthIndex: M } } }
        const monthMap = { january: 0, february: 1, march: 2, april: 3, may: 4, june: 5, july: 6, august: 7, september: 8, october: 9, november: 10, december: 11 };

        for (const year in groupedData) {
            console.log(`[${accountId}] Processing Year: ${year} for Monthly Summaries`);
            finalMonthlySummaries[year] = {};
            for (const monthObj of groupedData[year]) {
                for (const month in monthObj) {
                    const activities = monthObj[month];
                    console.log(`[${accountId}]   Processing Month: ${month} (${activities.length} activities)`);
                    if (activities.length === 0) {
                        console.log(`[${accountId}]   Skipping empty month: ${month} ${year}.`);
                        continue;
                    }

                    const monthIndex = monthMap[month.toLowerCase()];
                    if (monthIndex === undefined) {
                         console.warn(`[${accountId}]   Could not map month name: ${month}. Skipping.`);
                        continue;
                    }
                    // Use UTC to avoid timezone shifts when creating the start date
                    const startDate = new Date(Date.UTC(year, monthIndex, 1));
                    // Replace placeholder in the user prompt template
                    const userPromptMonthly = userPromptMonthlyTemplate.replace('{{YearMonth}}', `${month} ${year}`);

                    // --- REMOVED ASSISTANT CREATION ---

                    // Call OpenAI Assistant to generate the monthly summary using the verified ID
                    const monthlySummaryResult = await generateSummary(
                        activities, // Pass the raw activities array for this month
                        openai,
                        verifiedMonthlyAssistantId, // <-- Use verified ID
                        userPromptMonthly,
                        monthlyFuncSchema // Pass the specific schema for the monthly function
                    );

                    // Store the structured result from the AI function call along with metadata
                    finalMonthlySummaries[year][month] = {
                        aiOutput: monthlySummaryResult, // Store the full object from AI
                        count: activities.length,
                        startdate: startDate.toISOString().split('T')[0], // Format as YYYY-MM-DD
                        year: parseInt(year), // Ensure year is number
                        monthIndex: monthIndex // Store index for quarter calculation
                    };
                    console.log(`[${accountId}]   Generated monthly summary for ${month} ${year}.`);
                }
            }
        }

        // 3. Save Monthly Summaries to Salesforce (if any were generated)
        const monthlyForSalesforce = {};
        for (const year in finalMonthlySummaries) {
             monthlyForSalesforce[year] = {};
             for (const month in finalMonthlySummaries[year]) {
                 const monthData = finalMonthlySummaries[year][month];
                 const aiSummary = monthData.aiOutput?.summary || ''; // Extract HTML part
                 monthlyForSalesforce[year][month] = {
                     summary: JSON.stringify(monthData.aiOutput), // Keep full JSON
                     summaryDetails: aiSummary,
                     count: monthData.count,
                     startdate: monthData.startdate
                 };
             }
        }

        if (Object.keys(monthlyForSalesforce).length > 0 && Object.values(monthlyForSalesforce).some(year => Object.keys(year).length > 0)) {
            console.log(`[${accountId}] Saving monthly summaries to Salesforce...`);
            await createTimileSummarySalesforceRecords(conn, monthlyForSalesforce, accountId, 'Monthly', summaryRecordsMap);
            console.log(`[${accountId}] Monthly summaries saved.`);
        } else {
             console.log(`[${accountId}] No monthly summaries generated to save.`);
        }


        // --- 4. Group Monthly Summaries by Quarter ---
        console.log(`[${accountId}] Grouping monthly summaries by quarter...`);
        const quarterlyInputGroups = {}; // Structure: { "YYYY-QX": [ monthlyAiOutput1, monthlyAiOutput2, ... ] }

        for (const year in finalMonthlySummaries) {
            for (const month in finalMonthlySummaries[year]) {
                const monthData = finalMonthlySummaries[year][month];
                const quarter = getQuarterFromMonthIndex(monthData.monthIndex);
                const quarterKey = `${year}-${quarter}`; // e.g., "2023-Q1"

                if (!quarterlyInputGroups[quarterKey]) {
                    quarterlyInputGroups[quarterKey] = [];
                }
                // Push the actual AI output object needed for the quarterly prompt
                quarterlyInputGroups[quarterKey].push(monthData.aiOutput);
            }
        }
        console.log(`[${accountId}] Identified ${Object.keys(quarterlyInputGroups).length} quarters with data.`);


        // --- 5. Generate Quarterly Summary for EACH Quarter ---
        const allQuarterlyRawResults = {}; // Store raw AI output for each quarter { "YYYY-QX": quarterlyAiOutput }

        for (const [quarterKey, monthlySummariesForQuarter] of Object.entries(quarterlyInputGroups)) {
            console.log(`[${accountId}] Generating quarterly summary for ${quarterKey} using ${monthlySummariesForQuarter.length} monthly summaries...`);

             if (!monthlySummariesForQuarter || monthlySummariesForQuarter.length === 0) {
                console.warn(`[${accountId}] Skipping ${quarterKey} as it has no associated monthly summaries.`);
                continue;
            }

            // Prepare prompt with the specific monthly summaries for THIS quarter
            const quarterlyInputDataString = JSON.stringify(monthlySummariesForQuarter, null, 2);
            const [year, quarter] = quarterKey.split('-');
            const userPromptQuarterly = `${userPromptQuarterlyTemplate.replace('{{Quarter}}', quarter).replace('{{Year}}', year)}\n\nAggregate the following monthly summary data provided below for ${quarterKey}:\n\`\`\`json\n${quarterlyInputDataString}\n\`\`\``;

            // --- REMOVED ASSISTANT CREATION ---

            // Call AI using the verified quarterly assistant ID
            try {
                 const quarterlySummaryResult = await generateSummary(
                    null, // No raw activities needed
                    openai,
                    verifiedQuarterlyAssistantId, // <-- Use verified ID
                    userPromptQuarterly, // Prompt now contains the monthly summary JSON data
                    quarterlyFuncSchema // Pass the quarterly schema
                 );
                 allQuarterlyRawResults[quarterKey] = quarterlySummaryResult; // Store the raw AI JSON output
                 console.log(`[${accountId}] Successfully generated quarterly summary for ${quarterKey}.`);

            } catch (quarterlyError) {
                 console.error(`[${accountId}] Failed to generate quarterly summary for ${quarterKey}:`, quarterlyError);
                 // Log and continue. Consider how to report partial failures in the final callback.
            }
        }


        // --- 6. Transform and Consolidate ALL Quarterly Results ---
        console.log(`[${accountId}] Transforming ${Object.keys(allQuarterlyRawResults).length} generated quarterly summaries...`);
        const finalQuarterlyDataForSalesforce = {};

        for (const [quarterKey, rawAiResult] of Object.entries(allQuarterlyRawResults)) {
             const transformedResult = transformQuarterlyStructure(rawAiResult); // Process one quarter's AI output

             // Merge this single-quarter result into the final structure
             for (const year in transformedResult) {
                 if (!finalQuarterlyDataForSalesforce[year]) {
                     finalQuarterlyDataForSalesforce[year] = {};
                 }
                 for (const quarter in transformedResult[year]) {
                     // Basic check to avoid overwriting if the same quarter key appears unexpectedly
                     if (!finalQuarterlyDataForSalesforce[year][quarter]) {
                        finalQuarterlyDataForSalesforce[year][quarter] = transformedResult[year][quarter];
                     } else {
                         console.warn(`[${accountId}] Duplicate transformed data found for ${quarter} ${year}. Overwriting is prevented, but check transformation/grouping logic.`);
                     }
                 }
             }
        }


        // --- 7. Save ALL Generated Quarterly Summaries to Salesforce ---
         if (Object.keys(finalQuarterlyDataForSalesforce).length > 0 && Object.values(finalQuarterlyDataForSalesforce).some(year => Object.keys(year).length > 0)) {
            const totalQuarterlyRecords = Object.values(finalQuarterlyDataForSalesforce).reduce((sum, year) => sum + Object.keys(year).length, 0);
            console.log(`[${accountId}] Saving ${totalQuarterlyRecords} quarterly summaries to Salesforce...`);
            await createTimileSummarySalesforceRecords(conn, finalQuarterlyDataForSalesforce, accountId, 'Quarterly', summaryRecordsMap);
            console.log(`[${accountId}] Quarterly summaries saved.`);
        } else {
             console.log(`[${accountId}] No quarterly summaries generated or transformed to save.`);
        }


        // --- 8. Send Success Callback ---
        // TODO: Enhance status message if partial failures occurred (e.g., some quarters failed AI generation).
        console.log(`[${accountId}] Process completed.`);
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Success", "Summary Processed Successfully");

    } catch (error) {
        // Catch errors from any step (fetch, AI calls, save, transform)
        console.error(`[${accountId}] Error during summary processing:`, error);
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Processing error: ${error.message}`);
    }
}


// --- OpenAI Summary Generation Function ---
// No changes needed here - it already accepts assistantId dynamically.
async function generateSummary(
    activities,
    openaiClient,
    assistantId, // Accepts the ID dynamically (now passed from processSummary)
    userPrompt,
    functionSchema
) {
    let fileId = null;
    let thread = null;
    let filePath = null;
    let inputMethod = "prompt";

    // Ensure TEMP_FILE_DIR exists
    await fs.ensureDir(TEMP_FILE_DIR);

    try {
        thread = await openaiClient.beta.threads.create();
        console.log(`[Thread ${thread.id}] Created for Assistant ${assistantId}`);

        let finalUserPrompt = userPrompt;
        let messageAttachments = [];

        if (activities && Array.isArray(activities) && activities.length > 0) {
            let potentialFullPrompt;
            let activitiesJsonString;
            try {
                 activitiesJsonString = JSON.stringify(activities, null, 2);
                 potentialFullPrompt = `${userPrompt}\n\nHere is the activity data to process:\n\`\`\`json\n${activitiesJsonString}\n\`\`\``;
                 console.log(`[Thread ${thread.id}] Potential prompt length with direct JSON: ${potentialFullPrompt.length} characters.`);
            } catch(stringifyError) {
                console.error(`[Thread ${thread.id}] Error stringifying activities for length check:`, stringifyError);
                throw new Error("Failed to stringify activity data for processing.");
            }

            if (potentialFullPrompt.length < PROMPT_LENGTH_THRESHOLD && activities.length <= DIRECT_INPUT_THRESHOLD) {
                inputMethod = "direct JSON";
                finalUserPrompt = potentialFullPrompt;
                console.log(`[Thread ${thread.id}] Using direct JSON input (Prompt length ${potentialFullPrompt.length} < ${PROMPT_LENGTH_THRESHOLD}, Activities ${activities.length} <= ${DIRECT_INPUT_THRESHOLD}).`);
            } else {
                inputMethod = "file upload";
                console.log(`[Thread ${thread.id}] Using file upload (Prompt length ${potentialFullPrompt.length} >= ${PROMPT_LENGTH_THRESHOLD} or Activities ${activities.length} > ${DIRECT_INPUT_THRESHOLD}).`);
                finalUserPrompt = userPrompt; // Use original prompt when uploading file

                // --- Convert activities to Plain Text for file_search compatibility ---
                let activitiesText = activities.map((activity, index) => {
                    let activityLines = [`Activity ${index + 1}:`];
                    for (const [key, value] of Object.entries(activity)) {
                         // Handle null/undefined values gracefully
                        let displayValue = value === null || value === undefined ? 'N/A' :
                                           typeof value === 'object' ? JSON.stringify(value) : String(value);
                        activityLines.push(`  ${key}: ${displayValue}`);
                    }
                    return activityLines.join('\n');
                }).join('\n\n---\n\n');
                // ---------------------------------------------------------------------

                const timestamp = new Date().toISOString().replace(/[:.-]/g, "_");
                // *** USE .txt EXTENSION ***
                const filename = `salesforce_activities_${timestamp}_${thread.id}.txt`; // Change to .txt
                filePath = path.join(TEMP_FILE_DIR, filename); // Use the temp dir

                // *** WRITE THE TEXT STRING ***
                await fs.writeFile(filePath, activitiesText);
                console.log(`[Thread ${thread.id}] Temporary text file generated: ${filePath}`);

                const uploadResponse = await openaiClient.files.create({
                    file: fs.createReadStream(filePath),
                    purpose: "assistants",
                });
                fileId = uploadResponse.id;
                console.log(`[Thread ${thread.id}] File uploaded to OpenAI: ${fileId}`);

                // *** Attach file using the file_search tool type ***
                messageAttachments.push({ file_id: fileId, tools: [{ type: "file_search" }] });
                console.log(`[Thread ${thread.id}] Attaching file ${fileId} with file_search tool.`);
            }
        } else {
             console.log(`[Thread ${thread.id}] No activities array provided or array is empty. Using prompt content as is.`);
        }

        const messagePayload = {
            role: "user",
            content: finalUserPrompt,
        };
        if (messageAttachments.length > 0) {
            messagePayload.attachments = messageAttachments;
        }

        const message = await openaiClient.beta.threads.messages.create(thread.id, messagePayload);
        console.log(`[Thread ${thread.id}] Message added (using ${inputMethod}). ID: ${message.id}`);

        console.log(`[Thread ${thread.id}] Starting run, forcing function: ${functionSchema.name}`);
        // Ensure the assistant has 'file_search' enabled if using file uploads.
        // The 'tools' parameter here overrides assistant-level tools *for this run*,
        // forcing the specific function call. It does *not* disable file_search if the assistant has it.
        const run = await openaiClient.beta.threads.runs.createAndPoll(thread.id, {
            assistant_id: assistantId,
            // Specify the function tool for this run's primary purpose
            tools: [{ type: "function", function: functionSchema }],
            // Force the assistant to use this specific function
            tool_choice: { type: "function", function: { name: functionSchema.name } },
        });
        console.log(`[Thread ${thread.id}] Run status: ${run.status}`);

        if (run.status === 'requires_action') {
            const toolCalls = run.required_action?.submit_tool_outputs?.tool_calls;
            if (!toolCalls || toolCalls.length === 0) {
                 console.error(`[Thread ${thread.id}] Run requires action, but tool call data is missing or empty.`, run);
                 throw new Error("Function call was expected but not provided correctly by the Assistant.");
             }
             const toolCall = toolCalls[0]; // Assuming one function call as forced
             if (toolCall.function.name !== functionSchema.name) {
                  console.error(`[Thread ${thread.id}] Assistant called the wrong function. Expected: ${functionSchema.name}, Got: ${toolCall.function.name}`);
                  throw new Error(`Assistant called the wrong function: ${toolCall.function.name}`);
             }
             const rawArgs = toolCall.function.arguments;
             console.log(`[Thread ${thread.id}] Function call arguments received for ${toolCall.function.name}. Raw (truncated): ${rawArgs.substring(0,200)}...`);
             try {
                 const summaryObj = JSON.parse(rawArgs);
                 console.log(`[Thread ${thread.id}] Successfully parsed function arguments.`);
                 // NOTE: We are *not* submitting tool outputs back because the goal is just to get the structured data.
                 return summaryObj;
             } catch (parseError) {
                 console.error(`[Thread ${thread.id}] Failed to parse function call arguments JSON:`, parseError);
                 console.error(`[Thread ${thread.id}] Raw arguments received:`, rawArgs);
                 throw new Error(`Failed to parse function call arguments from AI: ${parseError.message}`);
             }
         } else if (run.status === 'completed') {
              console.warn(`[Thread ${thread.id}] Run completed without requiring function call action, despite tool_choice forcing ${functionSchema.name}. This might indicate an issue with the Assistant's setup or the prompt preventing function use.`);
              const messages = await openaiClient.beta.threads.messages.list(run.thread_id, { limit: 1 });
              const lastMessageContent = messages.data[0]?.content[0]?.text?.value || "No text content found.";
              console.warn(`[Thread ${thread.id}] Last message content from Assistant: ${lastMessageContent}`);
              // Consider if this should be an error or just a warning depending on expected behavior.
              // Throwing an error seems appropriate as the function call was mandated.
              throw new Error(`Assistant run completed without making the required function call to ${functionSchema.name}.`);
         } else {
             console.error(`[Thread ${thread.id}] Run failed or ended unexpectedly. Status: ${run.status}`, run.last_error);
             const errorMessage = run.last_error ? `${run.last_error.code}: ${run.last_error.message}` : 'Unknown error';
             throw new Error(`Assistant run failed. Status: ${run.status}. Error: ${errorMessage}`);
         }

    } catch (error) {
        console.error(`[Thread ${thread?.id || 'N/A'}] Error in generateSummary: ${error.message}`, error);
        throw error; // Re-throw
    } finally {
        // Cleanup temporary files and OpenAI files
        if (filePath) {
            try {
                await fs.unlink(filePath);
                console.log(`[Thread ${thread?.id || 'N/A'}] Deleted temporary file: ${filePath}`);
            } catch (unlinkError) {
                console.error(`[Thread ${thread?.id || 'N/A'}] Error deleting temporary file ${filePath}:`, unlinkError);
            }
        }
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
        // Optional: Delete the thread if desired (usually not necessary unless managing resource limits strictly)
        // if (thread) { try { await openaiClient.beta.threads.del(thread.id); } catch (e) { /* ignore */ } }
    }
}


// --- Salesforce Record Creation/Update Function ---
// No changes needed here, but ensure field names match your SF object
async function createTimileSummarySalesforceRecords(conn, summaries, parentId, summaryCategory, summaryRecordsMap) {
    console.log(`[${parentId}] Preparing to save ${summaryCategory} summaries...`);
    let recordsToCreate = [];
    let recordsToUpdate = [];

    for (const year in summaries) {
        for (const periodKey in summaries[year]) {
            const summaryData = summaries[year][periodKey];
            let summaryJsonString = summaryData.summaryJson || summaryData.summary; // Full AI JSON
            let summaryDetailsHtml = summaryData.summaryDetails || ''; // Extracted HTML
            let startDate = summaryData.startdate; // YYYY-MM-DD
            let count = summaryData.count;

             // Fallback to extract HTML from full JSON
             if (!summaryDetailsHtml && summaryJsonString) {
                try {
                    const parsedJson = JSON.parse(summaryJsonString);
                    // Adjust path based on actual structure (monthly vs quarterly)
                    if (summaryCategory === 'Monthly') {
                        summaryDetailsHtml = parsedJson?.summary || '';
                    } else if (summaryCategory === 'Quarterly') {
                        // The quarterly structure is nested differently
                         summaryDetailsHtml = parsedJson?.summary || ''; // Already extracted in transformQuarterlyStructure
                    }
                } catch (e) {
                    console.warn(`[${parentId}] Could not parse 'summaryJsonString' for ${periodKey} ${year} to extract HTML details.`);
                 }
            }

            let fyQuarterValue = (summaryCategory === 'Quarterly') ? periodKey : '';
            let monthValue = (summaryCategory === 'Monthly') ? periodKey : '';
            let shortMonth = monthValue ? monthValue.substring(0, 3) : '';
            let summaryMapKey = (summaryCategory === 'Quarterly') ? `${fyQuarterValue} ${year}` : `${shortMonth} ${year}`;
            let existingRecordId = getValueByKey(summaryRecordsMap, summaryMapKey);

            // --- Prepare Salesforce Record Payload ---
            // !!! DOUBLE CHECK THESE FIELD API NAMES !!!
            const recordPayload = {
                Parent_Id__c: parentId, // Lookup to Account or other parent
                Month__c: monthValue, // Text field
                Year__c: String(year), // Text or Number
                Summary_Category__c: summaryCategory, // Picklist ('Monthly', 'Quarterly')
                Summary__c: summaryJsonString ? summaryJsonString.substring(0, 131072) : null, // Long Text Area (131072 limit)
                Summary_Details__c: summaryDetailsHtml ? summaryDetailsHtml.substring(0, 131072) : null, // Rich Text Area (131072 limit)
                FY_Quarter__c: fyQuarterValue, // Text (e.g., 'Q1')
                Month_Date__c: startDate, // Date field
                Number_of_Records__c: count, // Number
                Account__c: parentId // Direct Lookup to Account (use if Parent_Id__c is not Account)
                // Add/remove fields as needed
            };

             if (!recordPayload.Parent_Id__c || !recordPayload.Summary_Category__c) {
                 console.warn(`[${parentId}] Skipping record for ${summaryMapKey} due to missing Parent ID or Category.`);
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

    // --- Perform Bulk DML Operations ---
    try {
        const options = { allOrNone: false }; // Process records independently

        if (recordsToCreate.length > 0) {
            console.log(`[${parentId}] Creating ${recordsToCreate.length} new ${summaryCategory} summary records via bulk API...`);
            const createResults = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "insert", options, recordsToCreate);
            handleBulkResults(createResults, recordsToCreate, 'create', parentId);
        } else {
            console.log(`[${parentId}] No new ${summaryCategory} records to create.`);
        }

        if (recordsToUpdate.length > 0) {
            console.log(`[${parentId}] Updating ${recordsToUpdate.length} existing ${summaryCategory} summary records via bulk API...`);
             const updateResults = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "update", options, recordsToUpdate);
             handleBulkResults(updateResults, recordsToUpdate, 'update', parentId);
        } else {
            console.log(`[${parentId}] No existing ${summaryCategory} records to update.`);
        }
    } catch (err) {
        console.error(`[${parentId}] Failed to save ${summaryCategory} records to Salesforce using Bulk API: ${err.message}`, err);
        throw new Error(`Salesforce save operation failed: ${err.message}`);
    }
}

// Helper to log bulk results
function handleBulkResults(results, originalPayloads, operationType, parentId) {
    console.log(`[${parentId}] Bulk ${operationType} results received (${results.length}).`);
    let successes = 0;
    let failures = 0;
    results.forEach((res, index) => {
        const recordIdentifier = originalPayloads[index].Id || `${originalPayloads[index].Month__c || originalPayloads[index].FY_Quarter__c} ${originalPayloads[index].Year__c}`;
        if (!res.success) {
            failures++;
            console.error(`[${parentId}] Error ${operationType} record ${index + 1} (${recordIdentifier}):`, res.errors);
        } else {
            successes++;
            // Optional: Log success
            // console.log(`[${parentId}] Successfully ${operationType}d record ${index + 1} (ID: ${res.id})`);
        }
    });
    console.log(`[${parentId}] Bulk ${operationType} summary: ${successes} succeeded, ${failures} failed.`);
}


// --- Salesforce Data Fetching with Pagination ---
// No changes needed here
async function fetchRecords(conn, queryOrUrl, allRecords = [], isFirstIteration = true) {
    try {
        const logPrefix = isFirstIteration ? `Initial Query (${(queryOrUrl || '').substring(0, 100)}...)` : "Fetching next batch";
        console.log(`[SF Fetch] ${logPrefix}`);

        const queryResult = isFirstIteration
            ? await conn.query(queryOrUrl)
            : await conn.queryMore(queryOrUrl);

        const fetchedCount = queryResult.records ? queryResult.records.length : 0;
        const currentTotal = allRecords.length + fetchedCount;
        console.log(`[SF Fetch] Fetched ${fetchedCount} records. Total so far: ${currentTotal}. Done: ${queryResult.done}`);

        if (fetchedCount > 0) {
            allRecords = allRecords.concat(queryResult.records);
        }

        if (!queryResult.done && queryResult.nextRecordsUrl) {
            return fetchRecords(conn, queryResult.nextRecordsUrl, allRecords, false);
        } else {
            console.log(`[SF Fetch] Finished fetching. Total records retrieved: ${allRecords.length}. Grouping...`);
            return groupRecordsByMonthYear(allRecords);
        }
    } catch (error) {
        console.error(`[SF Fetch] Error fetching Salesforce activities: ${error.message}`, error);
        throw error;
    }
}


// --- Data Grouping Helper Function ---
// No changes needed here
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
            // Extract only needed fields
            monthEntry[month].push({
                Id: activity.Id,
                Description: activity.Description || null, // Handle missing descriptions
                Subject: activity.Subject || null,       // Handle missing subjects
                ActivityDate: activity.ActivityDate     // Keep original date string
                // Add any other fields required by the AI function schemas
            });
        } catch(dateError) {
             console.warn(`Skipping activity (ID: ${activity.Id || 'Unknown'}) due to date processing error: ${dateError.message}. Date value: ${activity.ActivityDate}`);
        }
    });
    console.log("Finished grouping records by year and month.");
    return groupedData;
}


// --- Callback Sending Function ---
// No changes needed here
async function sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, status, message) {
    const logMessage = message.length > 200 ? message.substring(0, 200) + '...' : message;
    console.log(`[${accountId}] Sending callback to ${callbackUrl}. Status: ${status}, Message: ${logMessage}`);
    try {
        await axios.post(callbackUrl,
            {
                accountId: accountId,
                loggedinUserId: loggedinUserId,
                status: "Completed", // Callback status itself
                processResult: status, // 'Success' or 'Failed'
                message: message
            },
            {
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${accessToken}`
                },
                timeout: 30000 // Increased timeout for callback
            }
        );
        console.log(`[${accountId}] Callback sent successfully.`);
    } catch (error) {
        let errorMessage = `Failed to send callback to ${callbackUrl}. `;
        if (error.response) {
            errorMessage += `Status: ${error.response.status}, Data: ${JSON.stringify(error.response.data)}, Message: ${error.message}`;
        } else if (error.request) {
            errorMessage += `No response received. ${error.message}`;
        } else {
            errorMessage += `Error: ${error.message}`;
        }
        console.error(`[${accountId}] ${errorMessage}`);
    }
}


// --- Utility Helper Functions ---

// Finds a value in an array of {key: ..., value: ...} objects
// No changes needed
function getValueByKey(recordsArray, searchKey) {
    if (!recordsArray || !Array.isArray(recordsArray)) return null;
    const record = recordsArray.find(item => item && item.key === searchKey);
    return record ? record.value : null;
}

// Transforms the AI's quarterly output (for one quarter) for Salesforce saving.
// No changes needed
function transformQuarterlyStructure(quarterlyAiOutput) {
    const result = {};
    if (!quarterlyAiOutput?.yearlySummary?.[0]?.quarters?.[0]) {
        console.warn("Invalid structure received from quarterly AI transform function:", quarterlyAiOutput);
        return result; // Return empty object if essential parts are missing
    }

    const yearData = quarterlyAiOutput.yearlySummary[0];
    const year = yearData.year;
    const quarterData = yearData.quarters[0];

    if (!year || !quarterData.quarter) {
         console.warn(`Invalid year or quarter identifier in AI output for transform:`, quarterlyAiOutput);
         return result;
    }

    result[year] = {}; // Initialize year

    const htmlSummary = quarterData.summary || '';
    // Stringify the *quarterData* part, which contains the relevant details
    const fullQuarterlyJson = JSON.stringify(quarterData);
    const activityCount = quarterData.activityCount || 0;
    // Ensure startDate is YYYY-MM-DD format
    let startDate = quarterData.startdate;
    if (!startDate || !/^\d{4}-\d{2}-\d{2}$/.test(startDate)) {
        console.warn(`[Transform] Missing or invalid startdate format in quarterly AI output for ${quarterData.quarter} ${year}. Calculating default.`);
        startDate = `${year}-${getQuarterStartMonth(quarterData.quarter)}-01`;
    }


    result[year][quarterData.quarter] = {
        summaryDetails: htmlSummary,
        summaryJson: fullQuarterlyJson,
        count: activityCount,
        startdate: startDate
    };

    return result; // Structure: { 2023: { Q1: { ...data... } } }
}

// Helper to get the starting month number for a quarter string ('Q1'-'Q4')
// No changes needed
function getQuarterStartMonth(quarter) {
    switch (String(quarter).toUpperCase()) {
        case 'Q1': return '01';
        case 'Q2': return '04';
        case 'Q3': return '07';
        case 'Q4': return '10';
        default:
            console.warn(`Invalid quarter identifier "${quarter}" provided to getQuarterStartMonth. Defaulting to Q1.`);
            return '01';
    }
}
