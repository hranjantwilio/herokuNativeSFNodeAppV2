/*
 * Enhanced Node.js Express application for generating Salesforce activity summaries using OpenAI Assistants.
 *
 * Features:
 * - Creates/Retrieves OpenAI Assistants on startup using environment variable IDs as preference.
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
const { OpenAI, NotFoundError } = require("openai");
const fs = require("fs-extra"); // Using fs-extra for promise-based file operations and JSON handling
const path = require("path");
const axios = require("axios");

// --- Load Environment Variables ---
dotenv.config();

// --- Configuration Constants ---
const SF_LOGIN_URL = process.env.SF_LOGIN_URL;
const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
// --- Assistant IDs are now PREFERRED, creation is fallback ---
const OPENAI_MONTHLY_ASSISTANT_ID_ENV = process.env.OPENAI_MONTHLY_ASSISTANT_ID;
const OPENAI_QUARTERLY_ASSISTANT_ID_ENV = process.env.OPENAI_QUARTERLY_ASSISTANT_ID;

const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";
const TIMELINE_SUMMARY_OBJECT_API_NAME = "Timeline_Summary__c"; // Salesforce object API name
const DIRECT_INPUT_THRESHOLD = 2000; // Max activities for direct JSON input in prompt
const PROMPT_LENGTH_THRESHOLD = 256000; // Character limit for direct prompt input
const TEMP_FILE_DIR = path.join(__dirname, 'temp_files'); // Directory for temporary files

// --- Environment Variable Validation (Essential Vars) ---
if (!SF_LOGIN_URL || !OPENAI_API_KEY) {
    console.error("FATAL ERROR: Missing required environment variables (SF_LOGIN_URL, OPENAI_API_KEY).");
    process.exit(1);
}

// --- Global Variables for Final Assistant IDs ---
// These will be populated during startup by createOrRetrieveAssistant
let monthlyAssistantId = null;
let quarterlyAssistantId = null;


// --- Default OpenAI Function Schemas ---
// These define the structure the AI is *expected* to return via the function call.
// These full schemas are passed during the RUN, not during Assistant creation.
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
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public'))); // Optional: For serving static files

// --- Helper Function to Create or Retrieve Assistant ---
/**
 * Attempts to retrieve an Assistant by the provided ID (from env var).
 * If not found or no ID provided, creates a new Assistant.
 * @param {OpenAI} openaiClient - The initialized OpenAI client.
 * @param {string|null} assistantIdEnvVar - The Assistant ID from environment variable (preferred).
 * @param {string} assistantName - Name for the Assistant (used if creating).
 * @param {string} assistantInstructions - Instructions for the Assistant (used if creating).
 * @param {Array<object>} assistantToolsConfig - Base tools configuration (e.g., [{ type: "file_search" }, { type: "function" }]). Specific function schemas are NOT passed here.
 * @param {string} assistantModel - Model name (e.g., "gpt-4o").
 * @returns {Promise<string>} The ID of the retrieved or created Assistant.
 * @throws {Error} If retrieval fails (non-404) or creation fails.
 */
async function createOrRetrieveAssistant(
    openaiClient,
    assistantIdEnvVar,
    assistantName,
    assistantInstructions,
    assistantToolsConfig, // Renamed for clarity
    assistantModel
) {
    if (assistantIdEnvVar) {
        console.log(`Attempting to retrieve Assistant "${assistantName}" using ID: ${assistantIdEnvVar}`);
        try {
            const retrievedAssistant = await openaiClient.beta.assistants.retrieve(assistantIdEnvVar);
            console.log(`Successfully retrieved existing Assistant "${retrievedAssistant.name}" with ID: ${retrievedAssistant.id}`);
            // Optional: Check if config matches expectations
            // if (retrievedAssistant.model !== assistantModel) { console.warn(`Retrieved assistant ${assistantName} uses model ${retrievedAssistant.model}, expected ${assistantModel}`); }
            // if (!retrievedAssistant.tools.some(tool => tool.type === 'function')) { console.warn(`Retrieved assistant ${assistantName} does not have function tool enabled.`); }
            // if (!retrievedAssistant.tools.some(tool => tool.type === 'file_search')) { console.warn(`Retrieved assistant ${assistantName} does not have file_search tool enabled.`); }
            return retrievedAssistant.id;
        } catch (error) {
            if (error instanceof NotFoundError) {
                console.warn(`Assistant with ID "${assistantIdEnvVar}" not found. Will proceed to create a new one for "${assistantName}".`);
            } else {
                console.error(`Error retrieving Assistant "${assistantName}" (ID: ${assistantIdEnvVar}):`, error);
                throw new Error(`Failed to retrieve Assistant ${assistantName}: ${error.message}`);
            }
        }
    } else {
        console.log(`No environment variable ID provided for Assistant "${assistantName}". Proceeding to create a new one.`);
    }

    // --- Create Assistant if retrieval failed or no ID was provided ---
    console.log(`Creating new Assistant: ${assistantName}...`);
    try {
        // Pass only the base tool types (file_search, function) during creation
        const newAssistant = await openaiClient.beta.assistants.create({
            name: assistantName,
            instructions: assistantInstructions,
            tools: assistantToolsConfig, // Use the passed base tool config
            model: assistantModel,
        });
        console.log(`Successfully created new Assistant "${newAssistant.name}" with ID: ${newAssistant.id}`);
        // Construct potential env var name dynamically for the warning message
        const envVarName = `OPENAI_${assistantName.toUpperCase().replace(/ /g, '_').replace('SALESFORCE_', '')}_ASSISTANT_ID`;
        console.warn(`--> IMPORTANT: Consider adding this ID to your .env file as ${envVarName}=${newAssistant.id} for future reuse.`);
        return newAssistant.id;
    } catch (creationError) {
        console.error(`Error creating Assistant "${assistantName}":`, creationError);
        throw new Error(`Failed to create Assistant ${assistantName}: ${creationError.message}`);
    }
}


// --- Server Startup ---
// Wrap startup in an async IIFE to allow await for assistant setup
(async () => {
    try {
        console.log("Initializing Assistants...");
        await fs.ensureDir(TEMP_FILE_DIR); // Ensure temp directory exists

        // Define base tool configuration needed for assistants
        // These enable the *capabilities*. The specific function *schema* is passed during the run.
        const assistantBaseTools = [
             { type: "file_search" }, // Enable file searching capability
             { type: "function" }     // Enable function calling capability
        ];

        // --- Setup Monthly Assistant ---
        monthlyAssistantId = await createOrRetrieveAssistant(
            openai,
            OPENAI_MONTHLY_ASSISTANT_ID_ENV,
            "Salesforce Monthly Summarizer",
            "You are an AI assistant specialized in analyzing raw Salesforce activity data for a single month and generating structured JSON summaries using the provided function 'generate_monthly_activity_summary'. Apply sub-theme segmentation within the activityMapping as described in the function schema. Focus on extracting key themes, tone, and recommended actions. Use file_search if data is provided as a file.",
            assistantBaseTools, // Pass base tool config
            OPENAI_MODEL
        );

        // --- Setup Quarterly Assistant ---
        quarterlyAssistantId = await createOrRetrieveAssistant(
            openai,
            OPENAI_QUARTERLY_ASSISTANT_ID_ENV,
            "Salesforce Quarterly Summarizer", // Corrected name
            "You are an AI assistant specialized in aggregating pre-summarized monthly Salesforce activity data (provided as JSON in the prompt) into a structured quarterly JSON summary for a specific quarter using the provided function 'generate_quarterly_activity_summary'. Consolidate insights and activity lists accurately based on the input monthly summaries.",
             assistantBaseTools, // Pass base tool config
             OPENAI_MODEL
        );

        // Ensure both IDs were successfully obtained
        if (!monthlyAssistantId || !quarterlyAssistantId) {
             throw new Error("Failed to obtain valid IDs for one or more Assistants during startup.");
        }

        // Start the Express server only after assistants are ready
        app.listen(PORT, () => {
            console.log("----------------------------------------------------");
            console.log(`Server running on port ${PORT}`);
            console.log(`Using OpenAI Model (for new Assistants): ${OPENAI_MODEL}`);
            console.log(`Using Monthly Assistant ID: ${monthlyAssistantId}`);
            console.log(`Using Quarterly Assistant ID: ${quarterlyAssistantId}`);
            console.log(`Direct JSON input threshold: ${DIRECT_INPUT_THRESHOLD} activities`);
            console.log(`Prompt length threshold for file upload: ${PROMPT_LENGTH_THRESHOLD} characters`);
            console.log(`Temporary file directory: ${TEMP_FILE_DIR}`);
            console.log("----------------------------------------------------");
        });

    } catch (startupError) {
        console.error("FATAL STARTUP ERROR:", startupError.message);
        process.exit(1); // Exit if assistant setup fails
    }
})();


// --- Main API Endpoint ---
app.post('/generatesummary', async (req, res) => {
    console.log("Received /generatesummary request");

    // --- Ensure Assistants are Ready ---
    if (!monthlyAssistantId || !quarterlyAssistantId) {
        console.error("Error: Assistants not initialized properly during startup.");
        return res.status(500).json({ error: "Internal Server Error: Assistants not ready." });
    }

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
        console.warn("Bad Request: Missing required parameters. accessToken : " + accessToken + " loggedinUserId : "+ loggedinUserId + " accountId : " + accountId + " callbackUrl : " + callbackUrl + " queryText : " + queryText + " userPrompt : " + userPrompt + " userPromptQtr : " + userPromptQtr);
        return res.status(400).send({ error: "Missing required parameters (accountId, callbackUrl, accessToken, queryText, userPrompt, userPromptQtr, loggedinUserId)" });
    }

    // --- Parse Optional JSON Inputs & Function Schemas Safely ---
    let summaryRecordsMap = {};
    // Find default schemas - these will be used unless overridden
    let monthlyFuncSchema = defaultFunctions.find(f => f.name === 'generate_monthly_activity_summary');
    let quarterlyFuncSchema = defaultFunctions.find(f => f.name === 'generate_quarterly_activity_summary');

    try {
        if (summaryMap) {
            summaryRecordsMap = Object.entries(JSON.parse(summaryMap)).map(([key, value]) => ({ key, value }));
        }
        // --- Override default schemas if valid custom schemas are provided ---
        if (monthJSON) {
            const customMonthSchema = JSON.parse(monthJSON);
            // Basic validation of the custom schema
            if (!customMonthSchema || typeof customMonthSchema !== 'object' || !customMonthSchema.name || !customMonthSchema.parameters) {
                throw new Error("Provided monthJSON schema is invalid or missing required 'name' or 'parameters' properties.");
            }
            // Optional: More rigorous validation if needed
            monthlyFuncSchema = customMonthSchema; // Use the custom schema
            console.log("Using custom monthly function schema from request.");
        }
         if (qtrJSON) {
            const customQtrSchema = JSON.parse(qtrJSON);
            // Basic validation
            if (!customQtrSchema || typeof customQtrSchema !== 'object' || !customQtrSchema.name || !customQtrSchema.parameters) {
                 throw new Error("Provided qtrJSON schema is invalid or missing required 'name' or 'parameters' properties.");
            }
            quarterlyFuncSchema = customQtrSchema; // Use the custom schema
            console.log("Using custom quarterly function schema from request.");
        }
    } catch (e) {
        console.error("Failed to parse JSON input from request body:", e);
        return res.status(400).send({ error: `Invalid JSON provided in summaryMap, monthJSON, or qtrJSON. ${e.message}` });
    }

     // --- Ensure Schemas are Available ---
     if (!monthlyFuncSchema || !quarterlyFuncSchema) {
         // This check ensures that either the defaults were found or valid custom schemas were parsed.
         console.error("FATAL: Function schemas could not be loaded or parsed correctly.");
         return res.status(500).send({ error: "Internal server error: Could not load function schemas."});
     }

    // --- Acknowledge Request (202 Accepted) ---
    res.status(202).json({ status: 'processing', message: 'Summary generation initiated. You will receive a callback.' });
    console.log(`Initiating summary processing for Account ID: ${accountId}`);

    // --- Start Asynchronous Processing ---
    // Pass the globally stored, verified/created assistant IDs and the final function schemas
    processSummary(
        accountId,
        accessToken,
        callbackUrl,
        userPrompt,
        userPromptQtr,
        queryText,
        summaryRecordsMap,
        loggedinUserId,
        monthlyFuncSchema, // Pass the final schema (default or custom)
        quarterlyFuncSchema, // Pass the final schema (default or custom)
        monthlyAssistantId, // Pass the ID obtained during startup
        quarterlyAssistantId // Pass the ID obtained during startup
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

// --- Asynchronous Summary Processing Logic ---
// Function signature now accepts the final assistant IDs and schemas
async function processSummary(
    accountId,
    accessToken,
    callbackUrl,
    userPromptMonthlyTemplate,
    userPromptQuarterlyTemplate,
    queryText,
    summaryRecordsMap,
    loggedinUserId,
    finalMonthlyFuncSchema, // Receive the final schema to use
    finalQuarterlyFuncSchema, // Receive the final schema to use
    finalMonthlyAssistantId, // Receive the final ID
    finalQuarterlyAssistantId // Receive the final ID
) {
    console.log(`[${accountId}] Starting processSummary using Monthly Asst: ${finalMonthlyAssistantId}, Quarterly Asst: ${finalQuarterlyAssistantId}`);

    const conn = new jsforce.Connection({
        instanceUrl: SF_LOGIN_URL,
        accessToken: accessToken,
        maxRequest: 5,
        version: '59.0' // Use a recent API version
    });

    try {
        // 1. Fetch Salesforce Records
        console.log(`[${accountId}] Fetching Salesforce records...`);
        const groupedData = await fetchRecords(conn, queryText);
        console.log(`[${accountId}] Fetched and grouped data by year/month. Total record count: ${Object.values(groupedData).flatMap(yearData => yearData.flatMap(monthObj => Object.values(monthObj)[0])).length}`);

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
                    const startDate = new Date(Date.UTC(year, monthIndex, 1));
                    const userPromptMonthly = userPromptMonthlyTemplate.replace('{{YearMonth}}', `${month} ${year}`);

                    // --- Call generateSummary with the final monthly assistant ID and schema ---
                    const monthlySummaryResult = await generateSummary(
                        activities,
                        openai,
                        finalMonthlyAssistantId, // Use the ID obtained at startup
                        userPromptMonthly,
                        finalMonthlyFuncSchema // Use the final determined schema
                    );

                    // Store results
                    finalMonthlySummaries[year][month] = {
                         aiOutput: monthlySummaryResult,
                         count: activities.length,
                         startdate: startDate.toISOString().split('T')[0],
                         year: parseInt(year),
                         monthIndex: monthIndex
                    };
                     console.log(`[${accountId}]   Generated monthly summary for ${month} ${year}.`);
                 }
            }
        }

        // 3. Save Monthly Summaries to Salesforce
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


        // 4. Group Monthly Summaries by Quarter
        console.log(`[${accountId}] Grouping monthly summaries by quarter...`);
        const quarterlyInputGroups = {}; // Structure: { "YYYY-QX": [ monthlyAiOutput1, monthlyAiOutput2, ... ] }
        for (const year in finalMonthlySummaries) {
            for (const month in finalMonthlySummaries[year]) {
                const monthData = finalMonthlySummaries[year][month];
                const quarter = getQuarterFromMonthIndex(monthData.monthIndex);
                const quarterKey = `${year}-${quarter}`;

                if (!quarterlyInputGroups[quarterKey]) {
                    quarterlyInputGroups[quarterKey] = [];
                }
                quarterlyInputGroups[quarterKey].push(monthData.aiOutput); // Push the AI output needed for aggregation
            }
        }
        console.log(`[${accountId}] Identified ${Object.keys(quarterlyInputGroups).length} quarters with data.`);


        // 5. Generate Quarterly Summary for EACH Quarter
        const allQuarterlyRawResults = {}; // Store raw AI output for each quarter { "YYYY-QX": quarterlyAiOutput }
        for (const [quarterKey, monthlySummariesForQuarter] of Object.entries(quarterlyInputGroups)) {
            console.log(`[${accountId}] Generating quarterly summary for ${quarterKey} using ${monthlySummariesForQuarter.length} monthly summaries...`);

             if (!monthlySummariesForQuarter || monthlySummariesForQuarter.length === 0) {
                console.warn(`[${accountId}] Skipping ${quarterKey} as it has no associated monthly summaries.`);
                continue;
            }

            const quarterlyInputDataString = JSON.stringify(monthlySummariesForQuarter, null, 2);
            const [year, quarter] = quarterKey.split('-');
            const userPromptQuarterly = `${userPromptQuarterlyTemplate.replace('{{Quarter}}', quarter).replace('{{Year}}', year)}\n\nAggregate the following monthly summary data provided below for ${quarterKey}:\n\`\`\`json\n${quarterlyInputDataString}\n\`\`\``;

            // --- Call generateSummary with the final quarterly assistant ID and schema ---
            try {
                 const quarterlySummaryResult = await generateSummary(
                    null, // No raw activities needed, data is in the prompt
                    openai,
                    finalQuarterlyAssistantId, // Use the ID obtained at startup
                    userPromptQuarterly,
                    finalQuarterlyFuncSchema // Use the final determined schema
                 );
                 allQuarterlyRawResults[quarterKey] = quarterlySummaryResult; // Store the raw AI JSON output
                 console.log(`[${accountId}] Successfully generated quarterly summary for ${quarterKey}.`);
            } catch (quarterlyError) {
                 console.error(`[${accountId}] Failed to generate quarterly summary for ${quarterKey}:`, quarterlyError);
                 // Log and continue. Consider how to report partial failures.
            }
        }


        // 6. Transform and Consolidate ALL Quarterly Results
        console.log(`[${accountId}] Transforming ${Object.keys(allQuarterlyRawResults).length} generated quarterly summaries...`);
        const finalQuarterlyDataForSalesforce = {}; // Structure: { year: { QX: { summaryDetails, summaryJson, count, startdate } } }
        for (const [quarterKey, rawAiResult] of Object.entries(allQuarterlyRawResults)) {
             const transformedResult = transformQuarterlyStructure(rawAiResult); // Process one quarter's AI output
             // Merge this single-quarter result into the final structure
             for (const year in transformedResult) {
                 if (!finalQuarterlyDataForSalesforce[year]) {
                     finalQuarterlyDataForSalesforce[year] = {};
                 }
                 for (const quarter in transformedResult[year]) {
                     if (!finalQuarterlyDataForSalesforce[year][quarter]) {
                        finalQuarterlyDataForSalesforce[year][quarter] = transformedResult[year][quarter];
                     } else {
                         console.warn(`[${accountId}] Duplicate transformed data found for ${quarter} ${year}. Overwriting is prevented, but check logic.`);
                     }
                 }
             }
        }


        // 7. Save ALL Generated Quarterly Summaries to Salesforce
         if (Object.keys(finalQuarterlyDataForSalesforce).length > 0 && Object.values(finalQuarterlyDataForSalesforce).some(year => Object.keys(year).length > 0)) {
            const totalQuarterlyRecords = Object.values(finalQuarterlyDataForSalesforce).reduce((sum, year) => sum + Object.keys(year).length, 0);
            console.log(`[${accountId}] Saving ${totalQuarterlyRecords} quarterly summaries to Salesforce...`);
            await createTimileSummarySalesforceRecords(conn, finalQuarterlyDataForSalesforce, accountId, 'Quarterly', summaryRecordsMap);
            console.log(`[${accountId}] Quarterly summaries saved.`);
        } else {
             console.log(`[${accountId}] No quarterly summaries generated or transformed to save.`);
        }


        // 8. Send Success Callback
        // TODO: Enhance status message for partial failures if needed.
        console.log(`[${accountId}] Process completed.`);
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Success", "Summary Processed Successfully");

    } catch (error) {
        console.error(`[${accountId}] Error during summary processing:`, error);
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Processing error: ${error.message}`);
    }
}


// --- OpenAI Summary Generation Function ---
// This function handles the core interaction with the OpenAI Assistant for a single summary task.
// It accepts the specific Assistant ID and the detailed function schema for the run.
async function generateSummary(
    activities, // Array of activities or null (if data is in prompt)
    openaiClient,
    assistantId, // The ID of the specific Assistant to use (monthly or quarterly)
    userPrompt,
    functionSchema // The detailed schema for the function to be called in THIS run
) {
    let fileId = null;
    let thread = null;
    let filePath = null;
    let inputMethod = "prompt";

    try {
        // Ensure TEMP_FILE_DIR exists before attempting to write
        await fs.ensureDir(TEMP_FILE_DIR);

        // 1. Create a new Thread for this interaction
        thread = await openaiClient.beta.threads.create();
        console.log(`[Thread ${thread.id}] Created for Assistant ${assistantId}`);

        let finalUserPrompt = userPrompt;
        let messageAttachments = [];

        // 2. Determine input method: direct prompt vs. file upload
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

            // Check if direct JSON input is feasible based on thresholds
            if (potentialFullPrompt.length < PROMPT_LENGTH_THRESHOLD && activities.length <= DIRECT_INPUT_THRESHOLD) {
                inputMethod = "direct JSON";
                finalUserPrompt = potentialFullPrompt;
                console.log(`[Thread ${thread.id}] Using direct JSON input (Prompt length ${potentialFullPrompt.length} < ${PROMPT_LENGTH_THRESHOLD}, Activities ${activities.length} <= ${DIRECT_INPUT_THRESHOLD}).`);
            } else {
                // Use file upload if thresholds are exceeded
                inputMethod = "file upload";
                console.log(`[Thread ${thread.id}] Using file upload (Prompt length ${potentialFullPrompt.length} >= ${PROMPT_LENGTH_THRESHOLD} or Activities ${activities.length} > ${DIRECT_INPUT_THRESHOLD}).`);
                finalUserPrompt = userPrompt; // Use original prompt when uploading file

                // Convert activities to Plain Text for better file_search compatibility
                let activitiesText = activities.map((activity, index) => {
                    let activityLines = [`Activity ${index + 1}:`];
                    for (const [key, value] of Object.entries(activity)) {
                        let displayValue = value === null || value === undefined ? 'N/A' :
                                           typeof value === 'object' ? JSON.stringify(value) : String(value);
                        activityLines.push(`  ${key}: ${displayValue}`);
                    }
                    return activityLines.join('\n');
                }).join('\n\n---\n\n');

                // Create temporary file in the designated directory
                const timestamp = new Date().toISOString().replace(/[:.-]/g, "_");
                const filename = `salesforce_activities_${timestamp}_${thread.id}.txt`; // Use .txt extension
                filePath = path.join(TEMP_FILE_DIR, filename);

                await fs.writeFile(filePath, activitiesText);
                console.log(`[Thread ${thread.id}] Temporary text file generated: ${filePath}`);

                // Upload file to OpenAI
                const uploadResponse = await openaiClient.files.create({
                    file: fs.createReadStream(filePath),
                    purpose: "assistants", // Use 'assistants' purpose for Assistants API v2
                });
                fileId = uploadResponse.id;
                console.log(`[Thread ${thread.id}] File uploaded to OpenAI: ${fileId}`);

                // Attach file to the message using the file_search tool type
                messageAttachments.push({ file_id: fileId, tools: [{ type: "file_search" }] });
                console.log(`[Thread ${thread.id}] Attaching file ${fileId} with file_search tool.`);
            }
        } else {
             console.log(`[Thread ${thread.id}] No activities array provided or array is empty. Using prompt content as is.`);
        }

        // 3. Add the User Message to the Thread
        const messagePayload = { role: "user", content: finalUserPrompt };
        if (messageAttachments.length > 0) {
            messagePayload.attachments = messageAttachments;
        }
        const message = await openaiClient.beta.threads.messages.create(thread.id, messagePayload);
        console.log(`[Thread ${thread.id}] Message added (using ${inputMethod}). ID: ${message.id}`);

        // 4. Run the Assistant, providing the specific function schema and forcing its use
        console.log(`[Thread ${thread.id}] Starting run, forcing function: ${functionSchema.name}`);
        const run = await openaiClient.beta.threads.runs.createAndPoll(thread.id, {
            assistant_id: assistantId,
            // Pass the *detailed function schema* for THIS specific run.
            // This tells the assistant the exact structure of the function it can call now.
            tools: [{ type: "function", function: functionSchema }],
            // Force the assistant to use THIS specific function.
            tool_choice: { type: "function", function: { name: functionSchema.name } },
        });
        console.log(`[Thread ${thread.id}] Run status: ${run.status}`);

        // 5. Process the Run Outcome
        if (run.status === 'requires_action') {
            const toolCalls = run.required_action?.submit_tool_outputs?.tool_calls;
            if (!toolCalls || toolCalls.length === 0) {
                 console.error(`[Thread ${thread.id}] Run requires action, but tool call data is missing.`, run);
                 throw new Error("Function call was expected but not provided by the Assistant.");
             }
             // We forced a specific function, so expect only one tool call matching it
             const toolCall = toolCalls[0];
             if (toolCall.type !== 'function' || toolCall.function.name !== functionSchema.name) {
                  console.error(`[Thread ${thread.id}] Assistant required action for unexpected tool. Expected: ${functionSchema.name}, Got: ${toolCall.function?.name || toolCall.type}`);
                  throw new Error(`Assistant required action for unexpected tool: ${toolCall.function?.name || toolCall.type}`);
             }

             const rawArgs = toolCall.function.arguments;
             console.log(`[Thread ${thread.id}] Function call arguments received for ${toolCall.function.name}. Raw (truncated): ${rawArgs.substring(0, 200)}...`);
             try {
                 const summaryObj = JSON.parse(rawArgs);
                 console.log(`[Thread ${thread.id}] Successfully parsed function arguments.`);
                 // We return the parsed arguments directly as the desired output.
                 // No need to submit tool outputs back in this workflow.
                 return summaryObj;
             } catch (parseError) {
                 console.error(`[Thread ${thread.id}] Failed to parse function call arguments JSON:`, parseError);
                 console.error(`[Thread ${thread.id}] Raw arguments received:`, rawArgs);
                 throw new Error(`Failed to parse function call arguments from AI: ${parseError.message}`);
             }
         } else if (run.status === 'completed') {
              // This state is unexpected when tool_choice mandates a function call.
              console.warn(`[Thread ${thread.id}] Run completed without requiring function call action, despite tool_choice forcing ${functionSchema.name}. This might indicate an issue with the Assistant's setup or the prompt.`);
              const messages = await openaiClient.beta.threads.messages.list(run.thread_id, { limit: 1 });
              const lastMessageContent = messages.data[0]?.content[0]?.text?.value || "No text content found.";
              console.warn(`[Thread ${thread.id}] Last message content from Assistant: ${lastMessageContent}`);
              throw new Error(`Assistant run completed without making the required function call to ${functionSchema.name}.`);
         } else {
             // Handle other terminal statuses: failed, cancelled, expired
             console.error(`[Thread ${thread.id}] Run failed or ended unexpectedly. Status: ${run.status}`, run.last_error);
             const errorMessage = run.last_error ? `${run.last_error.code}: ${run.last_error.message}` : 'Unknown error';
             throw new Error(`Assistant run failed. Status: ${run.status}. Error: ${errorMessage}`);
         }

    } catch (error) {
        console.error(`[Thread ${thread?.id || 'N/A'}] Error in generateSummary: ${error.message}`, error);
        throw error; // Re-throw to be caught by the calling function (processSummary)
    } finally {
        // 6. Cleanup: Delete temporary file and OpenAI file
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
                 // Ignore 404 errors as the file might have been deleted already
                 if (!(deleteError instanceof NotFoundError || deleteError?.status === 404)) {
                    console.error(`[Thread ${thread?.id || 'N/A'}] Error deleting OpenAI file ${fileId}:`, deleteError.message || deleteError);
                 } else {
                     console.log(`[Thread ${thread?.id || 'N/A'}] OpenAI file ${fileId} already deleted or not found.`);
                 }
            }
        }
        // Optional: Delete the thread if resource management is critical
        // if (thread) { try { await openaiClient.beta.threads.del(thread.id); console.log(`[Thread ${thread.id}] Deleted thread.`); } catch (e) { /* ignore */ } }
    }
}


// --- Salesforce Record Creation/Update Function ---
// Uses Bulk API for efficiency
async function createTimileSummarySalesforceRecords(conn, summaries, parentId, summaryCategory, summaryRecordsMap) {
    console.log(`[${parentId}] Preparing to save ${summaryCategory} summaries...`);
    let recordsToCreate = [];
    let recordsToUpdate = [];

    // Iterate through the summaries structure { year: { periodKey: { summaryJson, summaryDetails, count, startdate } } }
    for (const year in summaries) {
        for (const periodKey in summaries[year]) { // periodKey is 'MonthName' or 'Q1', 'Q2' etc.
            const summaryData = summaries[year][periodKey];

            // Extract data
            let summaryJsonString = summaryData.summaryJson || summaryData.summary; // Full AI response JSON
            let summaryDetailsHtml = summaryData.summaryDetails || ''; // Extracted HTML summary
            let startDate = summaryData.startdate; // Should be YYYY-MM-DD
            let count = summaryData.count;

            // Fallback: Try to extract HTML from the full JSON if details field is empty/missing
             if (!summaryDetailsHtml && summaryJsonString) {
                try {
                    const parsedJson = JSON.parse(summaryJsonString);
                    // Structure differs slightly; 'summary' key holds HTML in both schemas here
                    summaryDetailsHtml = parsedJson?.summary || '';
                } catch (e) {
                    console.warn(`[${parentId}] Could not parse 'summaryJsonString' for ${periodKey} ${year} to extract HTML details.`);
                 }
            }

            // Determine Salesforce field values
            let fyQuarterValue = (summaryCategory === 'Quarterly') ? periodKey : '';
            let monthValue = (summaryCategory === 'Monthly') ? periodKey : '';
            let shortMonth = monthValue ? monthValue.substring(0, 3) : ''; // Abbreviated month for map key
            let summaryMapKey = (summaryCategory === 'Quarterly') ? `${fyQuarterValue} ${year}` : `${shortMonth} ${year}`;

            // Check if an existing record ID was provided
            let existingRecordId = getValueByKey(summaryRecordsMap, summaryMapKey);

            // --- Prepare Salesforce Record Payload ---
            // !!! CRITICAL: Verify these API names match your Salesforce object !!!
            const recordPayload = {
                // Use one Parent Id field - typically a direct lookup is better if always linking to Account
                Parent_Id__c: parentId, // Generic parent field (if used)
                //Account__c: parentId, // Direct lookup to Account (RECOMMENDED if always Account)
                Month__c: monthValue || null, // Text field for month name (null if quarterly)
                Year__c: String(year), // Text or Number field for year
                Summary_Category__c: summaryCategory, // Picklist ('Monthly', 'Quarterly')
                Summary__c: summaryJsonString ? summaryJsonString.substring(0, 131072) : null, // Long Text Area (check SF limit)
                Summary_Details__c: summaryDetailsHtml ? summaryDetailsHtml.substring(0, 131072) : null, // Rich Text Area (check SF limit)
                FY_Quarter__c: fyQuarterValue || null, // Text field for quarter (e.g., 'Q1') (null if monthly)
                Month_Date__c: startDate, // Date field for the start of the period
                Number_of_Records__c: count || 0, // Number field for activity count
                // Add other relevant fields like OwnerId, etc.
                // OwnerId: loggedinUserId // Example: Assign to the user initiating the request
            };

             // Basic validation before adding
             if (!recordPayload.Summary_Category__c || !recordPayload.Month_Date__c) {
                 console.warn(`[${parentId}] Skipping record for ${summaryMapKey} due to missing Category, or Start Date.`);
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
        const options = { allOrNone: false }; // Process records independently, allow partial success

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
        // Throw error to be caught by processSummary and trigger failure callback
        throw new Error(`Salesforce save operation failed: ${err.message}`);
    }
}

// Helper to log bulk API results
function handleBulkResults(results, originalPayloads, operationType, parentId) {
    console.log(`[${parentId}] Bulk ${operationType} results received (${results.length}).`);
    let successes = 0;
    let failures = 0;
    results.forEach((res, index) => {
        // Attempt to identify the record for logging purposes
        const recordIdentifier = originalPayloads[index].Id ||
                                 `${originalPayloads[index].Month__c || originalPayloads[index].FY_Quarter__c || 'N/A'} ${originalPayloads[index].Year__c || 'N/A'}`;
        if (!res.success) {
            failures++;
            // Log the specific error details from Salesforce
            console.error(`[${parentId}] Error ${operationType} record ${index + 1} (${recordIdentifier}): ${JSON.stringify(res.errors)}`);
        } else {
            successes++;
            // Optional: Log success ID
            // console.log(`[${parentId}] Successfully ${operationType}d record ${index + 1} (ID: ${res.id})`);
        }
    });
    console.log(`[${parentId}] Bulk ${operationType} summary: ${successes} succeeded, ${failures} failed.`);
}


// --- Salesforce Data Fetching with Pagination ---
// Recursively fetches all records for a given SOQL query using queryMore
async function fetchRecords(conn, queryOrUrl, allRecords = [], isFirstIteration = true) {
    try {
        const logPrefix = isFirstIteration ? `Initial Query (${(queryOrUrl || '').substring(0, 100)}...)` : "Fetching next batch";
        console.log(`[SF Fetch] ${logPrefix}`);

        // Use query() for the first call, queryMore() for subsequent calls with nextRecordsUrl
        const queryResult = isFirstIteration
            ? await conn.query(queryOrUrl)
            : await conn.queryMore(queryOrUrl); // queryOrUrl will be nextRecordsUrl here

        const fetchedCount = queryResult.records ? queryResult.records.length : 0;
        const currentTotal = allRecords.length + fetchedCount;
        console.log(`[SF Fetch] Fetched ${fetchedCount} records. Total so far: ${currentTotal}. Done: ${queryResult.done}`);

        if (fetchedCount > 0) {
            allRecords = allRecords.concat(queryResult.records);
        }

        // If queryResult indicates more records are available and provides a URL, fetch them recursively
        if (!queryResult.done && queryResult.nextRecordsUrl) {
            // Add a small delay to avoid hitting rate limits aggressively, especially with large datasets
            await new Promise(resolve => setTimeout(resolve, 200)); // 200ms delay
            return fetchRecords(conn, queryResult.nextRecordsUrl, allRecords, false);
        } else {
            // All records fetched, proceed to grouping
            console.log(`[SF Fetch] Finished fetching. Total records retrieved: ${allRecords.length}. Grouping...`);
            return groupRecordsByMonthYear(allRecords); // Group after all records are fetched
        }
    } catch (error) {
        console.error(`[SF Fetch] Error fetching Salesforce activities: ${error.message}`, error);
        // Consider specific error handling, e.g., for query timeouts or invalid queries
        throw error; // Re-throw to be handled by the caller (processSummary)
    }
}


// --- Data Grouping Helper Function ---
// Groups fetched Salesforce records by Year and then by Month Name
function groupRecordsByMonthYear(records) {
    const groupedData = {}; // { year: [ { MonthName: [activityObj, ...] }, ... ], ... }
    const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

    records.forEach(activity => {
        // Validate essential ActivityDate field
        if (!activity.ActivityDate) {
            console.warn(`Skipping activity (ID: ${activity.Id || 'Unknown'}) due to missing ActivityDate.`);
            return; // Skip record if date is missing
        }
        try {
            // Attempt to parse the date. Handle potential invalid date strings.
            const date = new Date(activity.ActivityDate);
             if (isNaN(date.getTime())) {
                 console.warn(`Skipping activity (ID: ${activity.Id || 'Unknown'}) due to invalid ActivityDate format: ${activity.ActivityDate}`);
                 return;
             }

            // Use UTC methods consistently to avoid timezone issues during grouping
            const year = date.getUTCFullYear();
            const monthIndex = date.getUTCMonth(); // 0-11
            const month = monthNames[monthIndex]; // Get month name

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
            // Select only necessary fields needed for the AI prompt/function
            monthEntry[month].push({
                Id: activity.Id,
                Description: activity.Description || null, // Use null for missing values if preferred
                Subject: activity.Subject || null,
                ActivityDate: activity.ActivityDate // Keep original format if needed elsewhere
                // Add other fields from the SOQL query if required by the AI prompt/function schemas
                // e.g., Type: activity.Type, Status: activity.Status
            });
        } catch(dateError) {
             // Catch potential errors during date processing (though basic validation is done above)
             console.warn(`Skipping activity (ID: ${activity.Id || 'Unknown'}) due to date processing error: ${dateError.message}. Date value: ${activity.ActivityDate}`);
        }
    });
    console.log("Finished grouping records by year and month.");
    return groupedData;
}


// --- Callback Sending Function ---
// Sends the final status back to the specified Salesforce URL
async function sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, status, message) {
    // Truncate long messages for logging clarity
    const logMessage = message.length > 500 ? message.substring(0, 500) + '...' : message;
    console.log(`[${accountId}] Sending callback to ${callbackUrl}. Status: ${status}, Message: ${logMessage}`);
    try {
        // Ensure status reflects the *process* outcome ('Success' or 'Failed')
        const processStatus = (status === "Success" || status === "Failed") ? status : "Failed"; // Default to Failed if status is unexpected

        await axios.post(callbackUrl,
            {
                // Payload structure expected by the callback receiver (e.g., an Apex REST service)
                accountId: accountId,
                loggedinUserId: loggedinUserId,
                status: "Completed", // Status of the *callback itself*
                processResult: processStatus, // Overall result ('Success' or 'Failed') of the summary generation
                message: message // Detailed message or error string
            },
            {
                headers: {
                    "Content-Type": "application/json",
                    // Use the provided session ID/access token for authenticating the callback request to Salesforce
                    "Authorization": `Bearer ${accessToken}`
                },
                timeout: 30000 // Increased timeout (30 seconds) for potentially slow callback endpoint
            }
        );
        console.log(`[${accountId}] Callback sent successfully.`);
    } catch (error) {
        let errorMessage = `Failed to send callback to ${callbackUrl}. `;
        if (error.response) {
            // Include response details from the callback endpoint if available
            errorMessage += `Status: ${error.response.status}, Data: ${JSON.stringify(error.response.data)}, Message: ${error.message}`;
        } else if (error.request) {
            // Error sending the request (e.g., network issue, DNS lookup failure)
            errorMessage += `No response received. ${error.message}`;
        } else {
            // Other errors (e.g., setting up the request)
            errorMessage += `Error: ${error.message}`;
        }
        console.error(`[${accountId}] ${errorMessage}`);
        // Depending on requirements, implement retry logic or log for manual intervention
    }
}


// --- Utility Helper Functions ---

// Finds a value in an array of {key: ..., value: ...} objects (used for summaryRecordsMap)
function getValueByKey(recordsArray, searchKey) {
    if (!recordsArray || !Array.isArray(recordsArray)) return null;
    const record = recordsArray.find(item => item && typeof item.key === 'string' && item.key.toLowerCase() === searchKey.toLowerCase()); // Case-insensitive search
    return record ? record.value : null; // Return the value or null if not found
}

// Transforms the AI's quarterly output structure (for a single quarter)
// into the format needed for Salesforce saving.
function transformQuarterlyStructure(quarterlyAiOutput) {
    const result = {}; // { year: { QX: { summaryDetails, summaryJson, count, startdate } } }

    // Add more robust validation for the expected nested structure
    if (!quarterlyAiOutput || typeof quarterlyAiOutput !== 'object' ||
        !Array.isArray(quarterlyAiOutput.yearlySummary) || quarterlyAiOutput.yearlySummary.length === 0 ||
        !quarterlyAiOutput.yearlySummary[0] || typeof quarterlyAiOutput.yearlySummary[0] !== 'object' ||
        !Array.isArray(quarterlyAiOutput.yearlySummary[0].quarters) || quarterlyAiOutput.yearlySummary[0].quarters.length === 0 ||
        !quarterlyAiOutput.yearlySummary[0].quarters[0] || typeof quarterlyAiOutput.yearlySummary[0].quarters[0] !== 'object')
    {
        console.warn("Invalid or incomplete structure received from quarterly AI for transformation:", JSON.stringify(quarterlyAiOutput));
        return result; // Return empty object if structure is significantly wrong
    }

    // Process the first year entry (quarterly AI should only return one year/quarter per call in this design)
    const yearData = quarterlyAiOutput.yearlySummary[0];
    const year = yearData.year;

    // Process the first quarter entry within that year
    const quarterData = yearData.quarters[0];
    const quarter = quarterData.quarter;

    // Validate essential quarter data
    if (!year || !quarter || typeof quarter !== 'string' || !quarter.match(/^Q[1-4]$/i)) {
         console.warn(`Invalid year or quarter identifier in quarterly AI output passed to transform: Year=${year}, Quarter=${quarter}`);
         return result;
    }

    // Extract the main HTML summary intended for display
    const htmlSummary = quarterData.summary || '';
    // Stringify the entire quarterData object (containing summary, mapping, count, etc.)
    // to store the full AI response context for this quarter.
    const fullQuarterlyJson = JSON.stringify(quarterData);
    // Get activity count, defaulting to 0 if missing or invalid
    const activityCount = (typeof quarterData.activityCount === 'number' && quarterData.activityCount >= 0) ? quarterData.activityCount : 0;
     // Get start date, calculating a default if missing or invalid format
    let startDate = quarterData.startdate;
    if (!startDate || typeof startDate !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(startDate)) {
        console.warn(`[Transform] Missing or invalid startdate format ('${startDate}') in quarterly AI output for ${quarter} ${year}. Calculating default.`);
        startDate = `${year}-${getQuarterStartMonth(quarter)}-01`;
    }

    // Initialize year object if not already present
    if (!result[year]) {
        result[year] = {};
    }

    // Structure the data for the createTimileSummarySalesforceRecords function
    result[year][quarter] = {
        summaryDetails: htmlSummary, // The extracted HTML summary
        summaryJson: fullQuarterlyJson, // The full JSON string of the quarter's data from AI
        count: activityCount, // Use a consistent 'count' key
        startdate: startDate // Use a consistent 'startdate' key (YYYY-MM-DD)
    };

    // Return structure like { 2023: { Q1: { ...data... } } }
    return result;
}

// Helper to get the starting month number (01, 04, 07, 10) for a quarter string ('Q1'-'Q4')
function getQuarterStartMonth(quarter) {
    if (!quarter || typeof quarter !== 'string') {
        console.warn(`Invalid quarter identifier "${quarter}" provided to getQuarterStartMonth. Defaulting to Q1.`);
        return '01';
    }
    switch (quarter.toUpperCase()) { // Ensure comparison is case-insensitive
        case 'Q1': return '01';
        case 'Q2': return '04';
        case 'Q3': return '07';
        case 'Q4': return '10';
        default:
            console.warn(`Unrecognized quarter identifier "${quarter}" provided to getQuarterStartMonth. Defaulting to Q1.`);
            return '01'; // Fallback to '01' for safety
    }
}
