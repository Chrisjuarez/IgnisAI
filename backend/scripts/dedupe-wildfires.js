#!/usr/bin/env node
/**
 * Collapse duplicate FIRMS detections.
 *
 * Before the write path became idempotent, every /api/wildfires call appended
 * a full snapshot, so the collection holds many copies of each detection. The
 * unique `firms_detection_identity` index cannot build until they are gone.
 *
 * Keeps the oldest document in each duplicate group and removes the rest.
 *
 * Usage:
 *   node backend/scripts/dedupe-wildfires.js            # dry run, reports only
 *   node backend/scripts/dedupe-wildfires.js --apply    # perform deletions
 */
const mongoose = require('mongoose');
const connectDB = require('../db');
const Wildfire = require('../models/Wildfire');

const DELETE_BATCH_SIZE = 1000;

function duplicateGroupsCursor() {
  return Wildfire.aggregate([
    {
      $group: {
        _id: {
          latitude: '$latitude',
          longitude: '$longitude',
          timestamp: '$timestamp',
          satellite: '$satellite',
          instrument: '$instrument',
        },
        ids: { $push: '$_id' },
        count: { $sum: 1 },
      },
    },
    { $match: { count: { $gt: 1 } } },
    { $project: { ids: 1 } },
  ])
    .allowDiskUse(true)
    .cursor({ batchSize: DELETE_BATCH_SIZE });
}

async function flush(ids, apply) {
  if (!apply || !ids.length) return;
  await Wildfire.deleteMany({ _id: { $in: ids } });
}

async function dedupe({ apply }) {
  let groups = 0;
  let redundant = 0;
  let pending = [];

  for await (const group of duplicateGroupsCursor()) {
    const [, ...duplicates] = group.ids;
    groups += 1;
    redundant += duplicates.length;
    pending.push(...duplicates);

    if (pending.length >= DELETE_BATCH_SIZE) {
      await flush(pending, apply);
      pending = [];
    }
  }

  await flush(pending, apply);
  return { groups, redundant };
}

async function main() {
  const apply = process.argv.includes('--apply');

  await connectDB();
  const before = await Wildfire.estimatedDocumentCount();
  const { groups, redundant } = await dedupe({ apply });

  console.log(`documents before      : ${before}`);
  console.log(`duplicated detections : ${groups}`);
  console.log(`redundant copies      : ${redundant}`);

  if (!apply) {
    console.log('\nDry run — nothing deleted. Re-run with --apply to remove them.');
    return;
  }

  console.log(`documents after       : ${await Wildfire.estimatedDocumentCount()}`);
  await Wildfire.syncIndexes();
  console.log('firms_detection_identity index built.');
}

main()
  .catch((err) => {
    console.error(`dedupe-wildfires failed: ${err.message}`);
    process.exitCode = 1;
  })
  .finally(() => mongoose.disconnect());
