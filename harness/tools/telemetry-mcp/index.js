#!/usr/bin/env node
const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} = require('@modelcontextprotocol/sdk/types.js');

const telemetry = require('./telemetry.js');

const server = new Server(
  { name: 'telemetry-mcp', version: '1.0.0' },
  { capabilities: { tools: {} } }
);

const TOOL = {
  name: 'recordTelemetry',
  description:
    'Append an event to telemetry.db (SQLite). Use for tool_call_start/end, workflow_step_start/end, state_change, error, and stage milestones (dev_started, qe_finished, harness_completed, etc).',
  inputSchema: {
    type: 'object',
    properties: {
      eventName: { type: 'string', description: 'Event name, e.g. dev_started' },
      details: { type: 'object', description: 'Arbitrary JSON details', additionalProperties: true },
    },
    required: ['eventName'],
  },
};

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: [TOOL] }));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  if (req.params.name !== 'recordTelemetry') {
    throw new Error(`Unknown tool: ${req.params.name}`);
  }
  const { eventName, details = {} } = req.params.arguments || {};
  if (!eventName) throw new Error('eventName required');
  telemetry.recordEvent(eventName, details);
  return {
    content: [{ type: 'text', text: `recorded: ${eventName}` }],
  };
});

(async () => {
  telemetry.getDb(); // eager-create telemetry.db + schema on first run
  const transport = new StdioServerTransport();
  await server.connect(transport);
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
