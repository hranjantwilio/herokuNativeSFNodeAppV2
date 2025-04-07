/*
 * Enhanced Node.js Express application for generating Salesforce activity summaries using OpenAI Assistants.
 *
 * Features:
 * - Creates/Retrieves OpenAI Assistants on startup and stores IDs.
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

const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";
const TIMELINE_SUMMARY_OBJECT_API_NAME = "Timeline_Summary__c"; // Salesforce object API name
const DIRECT_INPUT_THRESHOLD = 2000; // Max activities for direct JSON input in prompt
const TEMP_FILE_DIR = path.join(__dirname, 'temp_files'); // Directory for temporary files

// --- Environment Variable Validation ---
if (!SF_LOGIN_URL || !OPENAI_API_KEY) {
    console.error("FATAL ERROR: Missing required environment variables (SF_LOGIN_URL, OPENAI_API_KEY).");
    process.exit(1); // Exit if essential config is missing
}



// --- Default OpenAI Function Schemas ---
// These can be overridden by 'monthJSON' and 'qtrJSON' in the request body.
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
app.use(express.json({ limit: '10mb' })); // Increase JSON payload limit if needed for direct input
app.use(express.urlencoded({ extended: true, limit: '10mb' }));


// --- Helper to create an Assistant ---
async function createAssistant(name, instructions, tools, model) {

    // 2. Create a new assistant if no valid ID was provided or retrieval failed
    console.log(`Creating new Assistant: ${name}`);
    try {
        const newAssistant = await openai.beta.assistants.create({
            name: name,
            instructions: instructions,
            tools: tools,
            model: model,
        });
        console.log(`Successfully created new Assistant ${name} with ID: ${newAssistant.id}`);
        return newAssistant.id;
    } catch (creationError) {
         console.error(`Error creating Assistant ${name}:`, creationError);
         throw creationError; // Propagate error
    }
}


// --- Server Startup ---
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Using OpenAI Model: ${OPENAI_MODEL}`);
    console.log(`Direct JSON input threshold: ${DIRECT_INPUT_THRESHOLD} activities`);
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
    // Run processSummary without awaiting it here. Handle errors within the function or via catch block.
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
    monthlyFuncSchema,
    quarterlyFuncSchema
) {
    console.log(`[${accountId}] Starting processSummary...`);
    // Use the globally initialized Assistant IDs


    const conn = new jsforce.Connection({
        instanceUrl: SF_LOGIN_URL,
        accessToken: accessToken,
        // Consider adding maxRequest setting or handling token expiry for long processes
        // maxRequest: 10, // Example: retry requests up to 10 times
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
                    const monthlyAssistant = await openai.beta.assistants.create({
                        name: "Salesforce Monthly Summarizer",
                        instructions: "You are an AI assistant specialized in analyzing raw Salesforce activity data for a single month and generating structured JSON summaries using the provided function 'generate_monthly_activity_summary'. Apply sub-theme segmentation within the activityMapping as described in the function schema. Focus on extracting key themes, tone, and recommended actions.",
                        tools: [{ type: "file_search" },{type:"function" , "function" : monthlyFuncSchema}], // Allows using files
                        model: "gpt-4o",
                    });

                    // Call OpenAI Assistant to generate the monthly summary
                    const monthlySummaryResult = await generateSummary(
                        activities, // Pass the raw activities array for this month
                        openai,
                        monthlyAssistant.id, // Use global ID
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
        // Transform monthly data slightly for saving (extracting HTML summary etc.)
        const monthlyForSalesforce = {};
        for (const year in finalMonthlySummaries) {
             monthlyForSalesforce[year] = {};
             for (const month in finalMonthlySummaries[year]) {
                 const monthData = finalMonthlySummaries[year][month];
                 // Attempt to extract the main HTML summary, provide default if missing
                 const aiSummary = monthData.aiOutput?.summary || '';
                 monthlyForSalesforce[year][month] = {
                     summary: JSON.stringify(monthData.aiOutput), // Keep full JSON
                     summaryDetails: aiSummary, // Extracted HTML part
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

             // Check if monthlySummariesForQuarter is not empty before proceeding
            if (!monthlySummariesForQuarter || monthlySummariesForQuarter.length === 0) {
                console.warn(`[${accountId}] Skipping ${quarterKey} as it has no associated monthly summaries.`);
                continue;
            }

            // Prepare prompt with the specific monthly summaries for THIS quarter
            const quarterlyInputDataString = JSON.stringify(monthlySummariesForQuarter, null, 2);
            const [year, quarter] = quarterKey.split('-'); // Extract year and quarter ID
            // Make sure the quarterly prompt template clearly asks for aggregation of the provided JSON
            const userPromptQuarterly = `${userPromptQuarterlyTemplate.replace('{{Quarter}}', quarter).replace('{{Year}}', year)}\n\nAggregate the following monthly summary data provided below for ${quarterKey}:\n\`\`\`json\n${quarterlyInputDataString}\n\`\`\``;
            const quarterlyAssistant = await openai.beta.assistants.create({
                name: "Salesforce Quarterly Activities Summarizer",
                instructions: "You are an AI assistant specialized in aggregating pre-summarized monthly Salesforce activity data (provided as JSON in the prompt) into a structured quarterly JSON summary for a specific quarter using the provided function 'generate_quarterly_activity_summary'. Consolidate insights and activity lists accurately based on the input monthly summaries.",
                tools: [{ type: "file_search" },{type:"function" , "function" : quarterlyFuncSchema}], // Allows using files
                model: "gpt-4o",
            });
            // Call AI - Pass NULL for activities, data is in the prompt
            try {
                 const quarterlySummaryResult = await generateSummary(
                    null, // No raw activities needed
                    openai,
                    quarterlyAssistant.id, // Use global quarterly ID
                    userPromptQuarterly, // Prompt now contains the monthly summary JSON data
                    quarterlyFuncSchema // Pass the quarterly schema
                 );
                 allQuarterlyRawResults[quarterKey] = quarterlySummaryResult; // Store the raw AI JSON output
                 console.log(`[${accountId}] Successfully generated quarterly summary for ${quarterKey}.`);

            } catch (quarterlyError) {
                 console.error(`[${accountId}] Failed to generate quarterly summary for ${quarterKey}:`, quarterlyError);
                 // Log and continue to the next quarter. Consider adding quarterKey to failure callback later.
            }
        }


        // --- 6. Transform and Consolidate ALL Quarterly Results ---
        console.log(`[${accountId}] Transforming ${Object.keys(allQuarterlyRawResults).length} generated quarterly summaries...`);
        const finalQuarterlyDataForSalesforce = {}; // Structure: { year: { QX: { summaryDetails, summaryJson, count, startdate } } }

        for (const [quarterKey, rawAiResult] of Object.entries(allQuarterlyRawResults)) {
             // `transformQuarterlyStructure` expects the raw AI output format for a single quarter.
             // It should return { year: { QX: { ... } } } for that single quarter.
             const transformedResult = transformQuarterlyStructure(rawAiResult);

             // Merge this single-quarter result into the final structure for saving
             for (const year in transformedResult) {
                 if (!finalQuarterlyDataForSalesforce[year]) {
                     finalQuarterlyDataForSalesforce[year] = {};
                 }
                 for (const quarter in transformedResult[year]) {
                     if (!finalQuarterlyDataForSalesforce[year][quarter]) { // Ensure quarter doesn't overwrite if somehow processed twice
                        finalQuarterlyDataForSalesforce[year][quarter] = transformedResult[year][quarter];
                     } else {
                         console.warn(`[${accountId}] Duplicate transformed data found for ${quarter} ${year}. Overwriting is prevented, but check logic.`);
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
        // Consider if partial success needs a different message
        console.log(`[${accountId}] Process completed.`);
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Success", "Summary Processed Successfully"); // Adjust message if needed for partial failures

    } catch (error) {
        // Catch errors from any step (fetch, AI calls, save, transform)
        console.error(`[${accountId}] Error during summary processing:`, error);
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Processing error: ${error.message}`);
    }
}


// --- OpenAI Summary Generation Function ---
async function generateSummary(
    activities, // Array of raw activities OR null (if data is already embedded in prompt)
    openaiClient,
    assistantId, // Accepts the ID dynamically
    userPrompt, // The base user prompt template
    functionSchema // The specific function schema object for this call
) {
    let fileId = null;
    let thread = null;
    let filePath = null; // For temporary file path
    let inputMethod = "prompt"; // Default input method
    const PROMPT_LENGTH_THRESHOLD = 256000; // Define the character limit

    try {
        // Step 1: Create an OpenAI Thread
        thread = await openaiClient.beta.threads.create();
        console.log(`[Thread ${thread.id}] Created for Assistant ${assistantId}`);

        let finalUserPrompt = userPrompt; // Start with the base prompt
        let messageAttachments = []; // Attachments for the message (e.g., file IDs)

        // Step 2: Determine Input Method based on activities array AND prompt length
        if (activities && Array.isArray(activities) && activities.length > 0) {
            // --- Tentatively construct the prompt with direct JSON ---
            let potentialFullPrompt;
            let activitiesJsonString;
            try {
                 // Stringify first to check length accurately
                 activitiesJsonString = JSON.stringify(activities, null, 2); // Pretty print adds some length
                 potentialFullPrompt = `${userPrompt}\n\nHere is the activity data to process:\n\`\`\`json\n${activitiesJsonString}\n\`\`\``;
                 console.log(`[Thread ${thread.id}] Potential prompt length with direct JSON: ${potentialFullPrompt.length} characters.`);
            } catch(stringifyError) {
                console.error(`[Thread ${thread.id}] Error stringifying activities for length check:`, stringifyError);
                // Decide how to handle - perhaps default to file upload or throw
                throw new Error("Failed to stringify activity data for processing.");
            }


            // --- Check length against the threshold ---
            if (potentialFullPrompt.length < PROMPT_LENGTH_THRESHOLD && activities.length <= DIRECT_INPUT_THRESHOLD) {
                // --- Method: Direct JSON Input ---
                inputMethod = "direct JSON";
                finalUserPrompt = potentialFullPrompt; // Use the combined prompt
                console.log(`[Thread ${thread.id}] Using direct JSON input (Prompt length ${potentialFullPrompt.length} < ${PROMPT_LENGTH_THRESHOLD}).`);
                // No file upload needed, messageAttachments remains empty

            } else {
                // --- Method: File Upload Input (Prompt too long) ---
                inputMethod = "file upload";
                console.log(`[Thread ${thread.id}] Using file upload (Potential prompt length ${potentialFullPrompt.length} >= ${PROMPT_LENGTH_THRESHOLD}).`);
                // **IMPORTANT**: Use the *original* base userPrompt, not the one with JSON appended
                finalUserPrompt = userPrompt;

                 // --- Convert activities to Plain Text for file_search compatibility ---
                let activitiesText = activities.map((activity, index) => {
                    let activityLines = [`Activity ${index + 1}:`];
                    for (const [key, value] of Object.entries(activity)) {
                        let displayValue = typeof value === 'object' ? JSON.stringify(value) : value;
                        activityLines.push(`  ${key}: ${displayValue}`);
                    }
                    return activityLines.join('\n');
                }).join('\n\n---\n\n');
                // ---------------------------------------------------------------------

                // Create a temporary local file (.txt extension)
                const timestamp = new Date().toISOString().replace(/[:.-]/g, "_");
                // *** USE .txt EXTENSION ***
                const filename = `salesforce_activities_${timestamp}_${thread.id}.json`;
                filePath = path.join(__dirname, filename); // Or TEMP_FILE_DIR
                await fs.ensureDir(path.dirname(filePath)); // Ensure directory exists

                // *** WRITE THE TEXT STRING ***
                await fs.writeFile(filePath, activitiesText);
                console.log(`[Thread ${thread.id}] Temporary text file generated: ${filePath}`);

                // Upload the file to OpenAI
                const uploadResponse = await openaiClient.files.create({
                    file: fs.createReadStream(filePath),
                    purpose: "assistants", // Use 'assistants' purpose
                });
                fileId = uploadResponse.id;
                console.log(`[Thread ${thread.id}] File uploaded to OpenAI: ${fileId}`);

                // Prepare attachment for the message, linking the file and specifying the tool
                // *** Ensure file_search is used if that's the intent for large data ***
                messageAttachments.push({ file_id: fileId, tools: [{ type: "file_search" }] });
                console.log(`[Thread ${thread.id}] Attaching file ${fileId} with file_search tool.`);
            }
        } else {
             // Case where activities is null or empty (e.g., quarterly call where data is already in prompt)
             console.log(`[Thread ${thread.id}] No activities array provided or array is empty. Using prompt content as is.`);
             // finalUserPrompt is already set to userPrompt, messageAttachments is empty
        }

        // Step 3: Add the User Message to the Thread
        const messagePayload = {
            role: "user",
            content: finalUserPrompt, // Use the final prompt (either base or combined)
        };
        if (messageAttachments.length > 0) {
            messagePayload.attachments = messageAttachments;
        }

        const message = await openaiClient.beta.threads.messages.create(thread.id, messagePayload);
        console.log(`[Thread ${thread.id}] Message added (using ${inputMethod}). ID: ${message.id}`);

        // Step 4: Run the Assistant - CRITICAL: Force the specific function via tool_choice
        console.log(`[Thread ${thread.id}] Starting run, forcing function: ${functionSchema.name}`);
        const run = await openaiClient.beta.threads.runs.createAndPoll(thread.id, {
            assistant_id: assistantId,
            // IMPORTANT: Pass ONLY the required function schema in tools for this specific run
            // Note: If using file_search, the assistant might *also* need the file_search tool enabled here or at assistant level
            // Let's assume the assistant *already has* file_search enabled if needed. We only specify the function tool for this run's *primary* action.
            tools: [{ type: "function", function: functionSchema }],
            // Explicitly tell the Assistant it MUST call this specific function
            tool_choice: { type: "function", function: { name: functionSchema.name } },
        });
        console.log(`[Thread ${thread.id}] Run status: ${run.status}`);

        // Step 5: Process the Run Outcome (No changes needed here from previous version)
        if (run.status === 'requires_action') {
            const toolCalls = run.required_action?.submit_tool_outputs?.tool_calls;
            if (!toolCalls || toolCalls.length === 0) {
                 console.error(`[Thread ${thread.id}] Run requires action, but tool call data is missing or empty.`, run);
                 throw new Error("Function call was expected but not provided correctly by the Assistant.");
             }
             const toolCall = toolCalls[0];
             if (toolCall.function.name !== functionSchema.name) {
                  console.error(`[Thread ${thread.id}] Assistant called the wrong function. Expected: ${functionSchema.name}, Got: ${toolCall.function.name}`);
                  throw new Error(`Assistant called the wrong function: ${toolCall.function.name}`);
             }
             const rawArgs = toolCall.function.arguments;
             console.log(`[Thread ${thread.id}] Function call arguments received for ${toolCall.function.name}. Raw (truncated): ${rawArgs.substring(0,200)}...`);
             try {
                 const summaryObj = JSON.parse(rawArgs);
                 console.log(`[Thread ${thread.id}] Successfully parsed function arguments.`);
                 return summaryObj;
             } catch (parseError) {
                 console.error(`[Thread ${thread.id}] Failed to parse function call arguments JSON:`, parseError);
                 console.error(`[Thread ${thread.id}] Raw arguments received:`, rawArgs);
                 throw new Error(`Failed to parse function call arguments from AI: ${parseError.message}`);
             }
         } else if (run.status === 'completed') {
              console.warn(`[Thread ${thread.id}] Run completed without requiring function call action, despite tool_choice forcing ${functionSchema.name}.`);
              const messages = await openaiClient.beta.threads.messages.list(run.thread_id, { limit: 1 });
              const lastMessageContent = messages.data[0]?.content[0]?.text?.value || "No text content found.";
              console.warn(`[Thread ${thread.id}] Last message content from Assistant: ${lastMessageContent}`);
              throw new Error(`Assistant run completed without making the required function call to ${functionSchema.name}.`);
         } else {
             console.error(`[Thread ${thread.id}] Run failed or ended unexpectedly. Status: ${run.status}`, run.last_error);
             const errorMessage = run.last_error ? `${run.last_error.code}: ${run.last_error.message}` : 'Unknown error';
             throw new Error(`Assistant run failed. Status: ${run.status}. Error: ${errorMessage}`);
         }

    } catch (error) {
        console.error(`[Thread ${thread?.id || 'N/A'}] Error in generateSummary: ${error.message}`);
        throw error;
    } finally {
        // Step 6: Cleanup - Ensure temporary files and OpenAI files are deleted
        // *** Ensure correct filePath (which might be .txt now) is deleted ***
        if (filePath) {
            try {
                await fs.unlink(filePath);
                console.log(`[Thread ${thread?.id || 'N/A'}] Deleted temporary file: ${filePath}`);
            } catch (unlinkError) {
                console.error(`[Thread ${thread?.id || 'N/A'}] Error deleting temporary file ${filePath}:`, unlinkError);
            }
        }
        if (fileId) { // If a file was uploaded to OpenAI
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
    }
}

// // --- OpenAI Summary Generation Function ---
// async function generateSummary(
//     activities, // Array of raw activities OR null (if data is already embedded in prompt)
//     openaiClient,
//     assistantId, // Accepts the ID dynamically
//     userPrompt,
//     functionSchema // The specific function schema object for this call
// ) {
//     let fileId = null;
//     let thread = null;
//     let filePath = null; // For temporary file path
//     let inputMethod = "prompt"; // Default input method

//     try {
//         // Step 1: Create an OpenAI Thread
//         thread = await openaiClient.beta.threads.create();
//         console.log(`[Thread ${thread.id}] Created for Assistant ${assistantId}`);

//         let finalUserPrompt = userPrompt; // Base prompt
//         let messageAttachments = []; // Attachments for the message (e.g., file IDs)

//         // Step 2: Determine Input Method based on activities array
//         if (activities && Array.isArray(activities) && activities.length > 0) {
//             if (activities.length < DIRECT_INPUT_THRESHOLD) {
//                 // --- Method: Direct JSON Input ---
//                 inputMethod = "direct JSON";
//                 console.log(`[Thread ${thread.id}] Using direct JSON input (${activities.length} activities < ${DIRECT_INPUT_THRESHOLD}).`);
//                 const activitiesJsonString = JSON.stringify(activities, null, 2); // Pretty print for AI readability
//                 // Append the JSON data directly to the prompt content
//                 finalUserPrompt += `\n\nHere is the activity data to process:\n\`\`\`json\n${activitiesJsonString}\n\`\`\``;
//             } else {
//                 // --- Method: File Upload Input ---
//                 inputMethod = "file upload";
//                 console.log(`[Thread ${thread.id}] Using file upload (${activities.length} activities >= ${DIRECT_INPUT_THRESHOLD}).`);
//                 // Create a temporary local file
//                 const timestamp = new Date().toISOString().replace(/[:.-]/g, "_");
//                 const filename = `salesforce_activities_${timestamp}_${thread.id}.json`;
//                 //filePath = path.join(TEMP_FILE_DIR); // Store in the temp directory
//                 filePath = path.join(__dirname, filename);
//                 await fs.ensureDir(path.dirname(filePath)); // Ensure the temp directory exists
//                 await fs.writeJson(filePath, activities); // Write activities array as JSON
//                 console.log(`[Thread ${thread.id}] Temporary file generated: ${filePath}`);

//                 // Upload the file to OpenAI
//                 const uploadResponse = await openaiClient.files.create({
//                     file: fs.createReadStream(filePath),
//                     purpose: "assistants", // Use 'assistants' purpose
//                 });
//                 fileId = uploadResponse.id;
//                 console.log(`[Thread ${thread.id}] File uploaded to OpenAI: ${fileId}`);

//                 // Prepare attachment for the message, linking the file and specifying the tool
//                 messageAttachments.push({ file_id: fileId, tools: [{ type: "file_search" }] });
//             }
//         } else {
//              // Case where activities is null or empty (e.g., quarterly call)
//              console.log(`[Thread ${thread.id}] No activities array provided or array is empty. Using prompt content as is.`);
//         }

//         // Step 3: Add the User Message to the Thread
//         const messagePayload = {
//             role: "user",
//             content: finalUserPrompt, // Use the potentially modified prompt
//         };
//         if (messageAttachments.length > 0) {
//             messagePayload.attachments = messageAttachments;
//         }

//         const message = await openaiClient.beta.threads.messages.create(thread.id, messagePayload);
//         console.log(`[Thread ${thread.id}] Message added (using ${inputMethod}). ID: ${message.id}`);

//         // Step 4: Run the Assistant - CRITICAL: Force the specific function via tool_choice
//         console.log(`[Thread ${thread.id}] Starting run, forcing function: ${functionSchema.name}`);
//         const run = await openaiClient.beta.threads.runs.createAndPoll(thread.id, {
//             assistant_id: assistantId,
//             // IMPORTANT: Pass ONLY the required function schema in tools for this specific run
//             tools: [{ type: "function", function: functionSchema }],
//             // Explicitly tell the Assistant it MUST call this specific function
//             tool_choice: { type: "function", function: { name: functionSchema.name } },
//             // temperature: 0 // Optional: Set temperature to 0 for more deterministic output format
//         });
//         console.log(`[Thread ${thread.id}] Run status: ${run.status}`);

//         // Step 5: Process the Run Outcome
//         if (run.status === 'requires_action') {
//             // Check if the required action involves the expected tool call
//             const toolCalls = run.required_action?.submit_tool_outputs?.tool_calls;
//             if (!toolCalls || toolCalls.length === 0) {
//                 console.error(`[Thread ${thread.id}] Run requires action, but tool call data is missing or empty.`, run);
//                 throw new Error("Function call was expected but not provided correctly by the Assistant.");
//             }

//             // Assuming only one tool call is expected per run in this setup
//             const toolCall = toolCalls[0];

//             // Verify the correct function was called (important sanity check)
//             if (toolCall.function.name !== functionSchema.name) {
//                  console.error(`[Thread ${thread.id}] Assistant called the wrong function. Expected: ${functionSchema.name}, Got: ${toolCall.function.name}`);
//                  throw new Error(`Assistant called the wrong function: ${toolCall.function.name}`);
//             }

//             const rawArgs = toolCall.function.arguments;
//             console.log(`[Thread ${thread.id}] Function call arguments received for ${toolCall.function.name}. Raw (truncated): ${rawArgs.substring(0,200)}...`);
//             try {
//                 // Parse the JSON arguments returned by the function call
//                 const summaryObj = JSON.parse(rawArgs);
//                 console.log(`[Thread ${thread.id}] Successfully parsed function arguments.`);
//                 // We don't need to submit back tool outputs here, just return the parsed arguments
//                 return summaryObj;
//             } catch (parseError) {
//                 console.error(`[Thread ${thread.id}] Failed to parse function call arguments JSON:`, parseError);
//                 console.error(`[Thread ${thread.id}] Raw arguments received:`, rawArgs);
//                 throw new Error(`Failed to parse function call arguments from AI: ${parseError.message}`);
//             }
//         } else if (run.status === 'completed') {
//              // This is unexpected when tool_choice forces a function
//              console.warn(`[Thread ${thread.id}] Run completed without requiring function call action, despite tool_choice forcing ${functionSchema.name}. Check Assistant instructions or prompt clarity.`);
//              // Log the last message to understand what happened
//              const messages = await openaiClient.beta.threads.messages.list(run.thread_id, { limit: 1 });
//              const lastMessageContent = messages.data[0]?.content[0]?.text?.value || "No text content found.";
//              console.warn(`[Thread ${thread.id}] Last message content from Assistant: ${lastMessageContent}`);
//              throw new Error(`Assistant run completed without making the required function call to ${functionSchema.name}.`);
//         } else {
//             // Handle other terminal statuses: 'failed', 'cancelled', 'expired'
//             console.error(`[Thread ${thread.id}] Run failed or ended unexpectedly. Status: ${run.status}`, run.last_error);
//             const errorMessage = run.last_error ? `${run.last_error.code}: ${run.last_error.message}` : 'Unknown error';
//             throw new Error(`Assistant run failed. Status: ${run.status}. Error: ${errorMessage}`);
//         }

//     } catch (error) {
//         // Catch errors from thread creation, message sending, run execution, etc.
//         console.error(`[Thread ${thread?.id || 'N/A'}] Error in generateSummary: ${error.message}`);
//         throw error; // Re-throw the error to be caught by the calling function (processSummary)
//     } finally {
//         // Step 6: Cleanup - Ensure temporary files and OpenAI files are deleted
//         if (filePath) { // If a temporary local file was created
//             try {
//                 await fs.unlink(filePath);
//                 console.log(`[Thread ${thread?.id || 'N/A'}] Deleted temporary file: ${filePath}`);
//             } catch (unlinkError) {
//                 // Log error but don't necessarily fail the entire process
//                 console.error(`[Thread ${thread?.id || 'N/A'}] Error deleting temporary file ${filePath}:`, unlinkError);
//             }
//         }
//         if (fileId) { // If a file was uploaded to OpenAI
//             try {
//                 await openaiClient.files.del(fileId);
//                 console.log(`[Thread ${thread?.id || 'N/A'}] Deleted OpenAI file: ${fileId}`);
//             } catch (deleteError) {
//                 // Log error but continue; failing to delete shouldn't stop the summary process
//                  // Check if the error is ignorable (e.g., file already deleted)
//                  if (!(deleteError instanceof NotFoundError || deleteError?.status === 404)) {
//                     console.error(`[Thread ${thread?.id || 'N/A'}] Error deleting OpenAI file ${fileId}:`, deleteError.message || deleteError);
//                  } else {
//                      console.log(`[Thread ${thread?.id || 'N/A'}] OpenAI file ${fileId} already deleted or not found.`);
//                  }
//             }
//         }
//         // Optional: Delete thread? Usually not needed unless specifically managing resources.
//         // if (thread) { try { await openaiClient.beta.threads.del(thread.id); console.log(`[Thread ${thread.id}] Deleted thread.`); } catch (e) { /* log */ } }
//     }
// }


// --- Salesforce Record Creation/Update Function ---
async function createTimileSummarySalesforceRecords(conn, summaries, parentId, summaryCategory, summaryRecordsMap) {
    console.log(`[${parentId}] Preparing to save ${summaryCategory} summaries...`);
    let recordsToCreate = [];
    let recordsToUpdate = [];

    // Iterate through the summaries structure { year: { periodKey: { summaryJson, summaryDetails, count, startdate } } }
    for (const year in summaries) {
        for (const periodKey in summaries[year]) { // periodKey is 'MonthName' or 'Q1', 'Q2' etc.
            const summaryData = summaries[year][periodKey];

            // Extract data from the summary object for this period
            let summaryJsonString = summaryData.summaryJson || summaryData.summary; // Full AI response JSON
            let summaryDetailsHtml = summaryData.summaryDetails || ''; // Extracted HTML summary
            let startDate = summaryData.startdate; // Should be YYYY-MM-DD
            let count = summaryData.count;

            // Fallback: Try to extract HTML from the full JSON if details field is empty
             if (!summaryDetailsHtml && summaryJsonString) {
                try {
                    const parsedJson = JSON.parse(summaryJsonString);
                    // Adjust path based on the actual structure returned by your AI function schema
                    summaryDetailsHtml = parsedJson.summary || ''; // Assumes top-level 'summary' key holds HTML
                } catch (e) {
                    console.warn(`[${parentId}] Could not parse 'summaryJsonString' for ${periodKey} ${year} to extract HTML details. HTML field might be empty.`);
                    // Keep summaryDetailsHtml as empty string if parsing fails
                 }
            }

            // Determine Salesforce field values based on category
            let fyQuarterValue = (summaryCategory === 'Quarterly') ? periodKey : '';
            let monthValue = (summaryCategory === 'Monthly') ? periodKey : '';
            // Create a consistent key for looking up existing records
            let shortMonth = monthValue ? monthValue.substring(0, 3) : '';
            let summaryMapKey = (summaryCategory === 'Quarterly') ? `${fyQuarterValue} ${year}` : `${shortMonth} ${year}`;

            // Check if an existing record ID was provided for this period
            let existingRecordId = getValueByKey(summaryRecordsMap, summaryMapKey);

            // --- Prepare Salesforce Record Payload ---
            // IMPORTANT: Use the EXACT API names of your fields in Timeline_Summary__c
            const recordPayload = {
                Parent_Id__c: parentId, // Lookup to parent record (e.g., Account)
                Month__c: monthValue, // Text field for month name
                Year__c: String(year), // Text or Number field for year
                Summary_Category__c: summaryCategory, // Picklist ('Monthly', 'Quarterly')
                Summary__c: summaryJsonString ? summaryJsonString.substring(0, 131072) : null, // Long Text Area (check limit)
                Summary_Details__c: summaryDetailsHtml ? summaryDetailsHtml.substring(0, 131072) : null, // Rich Text Area (check limit)
                FY_Quarter__c: fyQuarterValue, // Text field for quarter (e.g., 'Q1')
                Month_Date__c: startDate, // Date field for the start of the period
                Number_of_Records__c: count, // Number field for activity count
                Account__c: parentId // Assuming Account lookup relationship field API name
            };

             // Basic validation before adding
             if (!recordPayload.Parent_Id__c || !recordPayload.Summary_Category__c) {
                 console.warn(`[${parentId}] Skipping record for ${summaryMapKey} due to missing Parent ID or Category.`);
                 continue;
             }

            // Add to appropriate list for bulk operation
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
        // Use allOrNone=false to allow partial success
        const options = { allOrNone: false };

        if (recordsToCreate.length > 0) {
            console.log(`[${parentId}] Creating ${recordsToCreate.length} new ${summaryCategory} summary records via bulk API...`);
            // Use bulk API for better performance with many records
            const createResults = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "insert", options, recordsToCreate);
            console.log(`[${parentId}] Bulk create results received (${createResults.length}).`);
            createResults.forEach((res, index) => {
                if (!res.success) {
                    console.error(`[${parentId}] Error creating record ${index + 1} (${recordsToCreate[index].Month__c || recordsToCreate[index].FY_Quarter__c} ${recordsToCreate[index].Year__c}):`, res.errors);
                } else {
                    // Optional: Log success
                    // console.log(`[${parentId}] Successfully created record ${index + 1} (ID: ${res.id})`);
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
                 } else {
                     // Optional: Log success
                     // console.log(`[${parentId}] Successfully updated record ${index + 1} (ID: ${res.id})`);
                 }
             });
        } else {
            console.log(`[${parentId}] No existing ${summaryCategory} records to update.`);
        }
    } catch (err) {
        console.error(`[${parentId}] Failed to save ${summaryCategory} records to Salesforce using Bulk API: ${err.message}`, err);
        // Throw error to be caught by processSummary and trigger failure callback
        throw new Error(`Salesforce save operation failed: ${err.message}`);
    }
}


// --- Salesforce Data Fetching with Pagination ---
async function fetchRecords(conn, queryOrUrl, allRecords = [], isFirstIteration = true) {
    try {
        console.log(`QUERY : ${queryOrUrl}. `);
        //const logPrefix = isFirstIteration ? "Initial query" : "Querying more records from";
        // Avoid logging potentially sensitive parts of the URL/query
        //console.log(`${logPrefix}: ${typeof queryOrUrl === 'string' && queryOrUrl.startsWith('SELECT') ? queryOrUrl.substring(0, 150) + '...' : 'nextRecordsUrl'}`);

        // Use query() for initial SOQL, queryMore() for subsequent nextRecordsUrl
        const queryResult = isFirstIteration
            ? await conn.query(queryOrUrl)
            : await conn.queryMore(queryOrUrl); // queryOrUrl here is nextRecordsUrl

        const fetchedCount = queryResult.records ? queryResult.records.length : 0;
        console.log(`Fetched 1000 more records. Done: ${queryResult.done}`);

        // if (fetchedCount > 0) {
        //     allRecords = allRecords.concat(queryResult.records);
        // }

        // Check if more records exist and a URL is provided
        if (!queryResult.done && queryResult.nextRecordsUrl) {
            // Recursively fetch the next batch
            return fetchRecords(conn, queryResult.nextRecordsUrl, allRecords, false);
        } else {
            // All records fetched, proceed to grouping
            console.log(`Finished fetching. Total records retrieved: ${allRecords.length}. Grouping...`);
            return groupRecordsByMonthYear(allRecords); // Group after all records are fetched
        }
    } catch (error) {
        console.error(`Error fetching Salesforce activities: ${error.message}`, error);
        throw error; // Re-throw to be handled by the caller (processSummary)
    }
}


// --- Data Grouping Helper Function ---
function groupRecordsByMonthYear(records) {
    const groupedData = {}; // { year: [ { MonthName: [activityObj, ...] }, ... ], ... }
    records.forEach(activity => {
        // Validate essential ActivityDate field
        if (!activity.ActivityDate) {
            console.warn(`Skipping activity (ID: ${activity.Id || 'Unknown'}) due to missing ActivityDate.`);
            return; // Skip record if date is missing
        }
        try {
            const date = new Date(activity.ActivityDate);
            // Check for invalid date object
             if (isNaN(date.getTime())) {
                 console.warn(`Skipping activity (ID: ${activity.Id || 'Unknown'}) due to invalid ActivityDate format: ${activity.ActivityDate}`);
                 return;
             }

            // Use UTC methods to avoid timezone interpretation issues during grouping
            const year = date.getUTCFullYear(); // Use UTC to avoid timezone issues
            const month = date.toLocaleString('en-US', { month: 'long', timeZone: 'UTC' }); // Get month name in UTC

            // Initialize year array if it doesn't exist
            if (!groupedData[year]) {
                groupedData[year] = [];
            }

            // Find or create the object for the specific month within the year's array
            let monthEntry = groupedData[year].find(entry => entry[month]);
            if (!monthEntry) {
                monthEntry = { [month]: [] };
                groupedData[year].push(monthEntry);
            }

            // Add the relevant activity details to the month's array
            // Select only necessary fields to reduce memory usage
            monthEntry[month].push({
                Id: activity.Id,
                Description: activity.Description || "No Description",
                Subject: activity.Subject || "No Subject",
                ActivityDate: activity.ActivityDate // Keep original format if needed
                // Add other fields from the query if required by the AI prompt/function
            });
        } catch(dateError) {
             // Catch potential errors during date processing
             console.warn(`Skipping activity (ID: ${activity.Id || 'Unknown'}) due to date processing error: ${dateError.message}. Date value: ${activity.ActivityDate}`);
        }
    });
    console.log("Finished grouping records by year and month.");
    return groupedData;
}


// --- Callback Sending Function ---
async function sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, status, message) {
    // Truncate long messages for logging
    const logMessage = message.length > 200 ? message.substring(0, 200) + '...' : message;
    console.log(`[${accountId}] Sending callback to ${callbackUrl}. Status: ${status}, Message: ${logMessage}`);
    try {
        await axios.post(callbackUrl,
            {
                // Payload structure expected by the callback receiver
                accountId: accountId,
                loggedinUserId: loggedinUserId,
                status: "Completed", // Status of the *callback sending action* itself
                processResult: status, // Overall result ('Success' or 'Failed') of the summary generation
                message: message // Detailed message or error
            },
            {
                headers: {
                    "Content-Type": "application/json",
                    // Use the provided access token for authenticating the callback request
                    "Authorization": `Bearer ${accessToken}`
                },
                timeout: 20000 // Set a reasonable timeout (e.g., 20 seconds)
            }
        );
        console.log(`[${accountId}] Callback sent successfully.`);
    } catch (error) {
        let errorMessage = error.message;
        if (error.response) {
            // Include response details if available (status code, data)
            errorMessage = `Status: ${error.response.status}, Data: ${JSON.stringify(error.response.data)}, Message: ${error.message}`;
        } else if (error.request) {
            errorMessage = `No response received from callback URL. ${error.message}`;
        }
        console.error(`[${accountId}] Failed to send callback to ${callbackUrl}: ${errorMessage}`);
        // Depending on requirements, you might implement retry logic here or log for manual follow-up
    }
}


// --- Utility Helper Functions ---

// Finds a value in an array of {key: ..., value: ...} objects
function getValueByKey(recordsArray, searchKey) {
    if (!recordsArray || !Array.isArray(recordsArray)) return null;
    const record = recordsArray.find(item => item && item.key === searchKey);
    return record ? record.value : null; // Return the value or null if not found
}

// Transforms the AI's quarterly output structure (for a single quarter)
// to the format needed for Salesforce saving.
function transformQuarterlyStructure(quarterlyAiOutput) {
    const result = {}; // { year: { QX: { summaryDetails, summaryJson, count, startdate } } }
    // Basic validation of the input structure from the AI
    if (!quarterlyAiOutput || !Array.isArray(quarterlyAiOutput.yearlySummary) || quarterlyAiOutput.yearlySummary.length === 0) {
        console.warn("Invalid or empty structure received from quarterly AI transform function:", quarterlyAiOutput);
        return result; // Return empty object if structure is wrong
    }

    // Process the first year entry (quarterly AI should only return one year/quarter per call)
    const yearData = quarterlyAiOutput.yearlySummary[0];
    if (!yearData || !yearData.year || !Array.isArray(yearData.quarters) || yearData.quarters.length === 0) {
        console.warn(`Invalid year or missing quarters data in quarterly AI output passed to transform:`, yearData);
        return result;
    }

    const year = yearData.year;
    result[year] = {}; // Initialize year object

    // Process the first quarter entry within that year (should only be one)
    const quarterData = yearData.quarters[0];
    if (!quarterData || !quarterData.quarter || !quarterData.summary) {
         console.warn(`Invalid quarter data in quarterly AI output passed to transform for year ${year}:`, quarterData);
         // Return the result potentially with just the empty year object if quarter is invalid
         return result;
    }

    // Extract the main HTML summary intended for display
    const htmlSummary = quarterData.summary || '';
    // Stringify the entire quarterData object to store the full AI response context
    const fullQuarterlyJson = JSON.stringify(quarterData);
    // Get activity count, defaulting to 0 if missing
    const activityCount = quarterData.activityCount || 0;
     // Get start date, calculating a default if missing
    const startDate = quarterData.startdate || `${year}-${getQuarterStartMonth(quarterData.quarter)}-01`;


    // Structure the data for the createTimileSummarySalesforceRecords function
    result[year][quarterData.quarter] = {
        summaryDetails: htmlSummary, // The extracted HTML summary
        summaryJson: fullQuarterlyJson, // The full JSON string of the quarter's data from AI
        count: activityCount, // Use a consistent 'count' key
        startdate: startDate // Use a consistent 'startdate' key
    };

    // Return structure like { 2023: { Q1: { ...data... } } }
    return result;
}

// Helper to get the starting month number (01, 04, 07, 10) for a quarter string ('Q1'-'Q4')
function getQuarterStartMonth(quarter) {
    switch (String(quarter).toUpperCase()) { // Ensure comparison is case-insensitive
        case 'Q1': return '01';
        case 'Q2': return '04';
        case 'Q3': return '07';
        case 'Q4': return '10';
        default:
            console.warn(`Invalid quarter identifier "${quarter}" provided to getQuarterStartMonth. Defaulting to Q1 start month.`);
            return '01'; // Fallback to '01' for safety
    }
}
