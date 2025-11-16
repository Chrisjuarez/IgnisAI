const path = require('path');
const request = require('supertest');
const mongoose = require('mongoose');

// ⬇️ Adjust these to match your project structure
const app = require('../app');                    // Express app
const Fire = require('../models/Fire');          // Active fire model
const Report = require('../models/Report');      // Crowdsourced report model
const storage = require('../services/storage');  // Media storage (S3, GCS, etc.)
const exifService = require('../services/exifService'); // EXIF stripping utility

// Mock external side-effect services so we can assert on them
jest.mock('../services/storage');
jest.mock('../services/exifService');

describe('Crowdsourced Reports Moderation & Privacy (High Priority)', () => {
  let activeFire;
  let userToken;
  let adminToken;

  beforeAll(async () => {

    // 🧪 Fake auth tokens – replace with your real test auth helpers
    userToken = 'test-user-token';   // Your auth middleware should accept this in test
    adminToken = 'test-admin-token'; // And recognize this as an admin
  });

  beforeEach(async () => {
    // Clear collections before each test
    await Report.deleteMany({});
    await Fire.deleteMany({});

    // Create an active fire record for attaching reports
    activeFire = await Fire.create({
      name: 'Test Fire',
      status: 'active',
      location: {
        type: 'Point',
        coordinates: [-118.25, 34.05],
      },
    });

    // Reset all mocks
    jest.clearAllMocks();

    // Mock EXIF stripping behavior:
    // given a buffer, return a "sanitized" buffer with EXIF removed
    exifService.stripExifMetadata.mockImplementation(async (buffer) => {
      // In a real impl, you'd use exiftool/sharp/etc.
      // For test, we just return a new Buffer to distinguish it
      return Buffer.from(buffer.toString().replace(/EXIF:.+?;/g, ''));
    });

    // Mock storage service (e.g., S3/GCS upload & delete)
    storage.savePhoto.mockResolvedValue({
      key: 'reports/photos/sanitized-photo.jpg',
      url: 'https://cdn.example.com/reports/photos/sanitized-photo.jpg',
    });

    storage.deletePhoto.mockResolvedValue(true);
  });

  afterAll(async () => {
    await mongoose.connection.close();
  });

  it('should enforce moderation, EXIF stripping, privacy, and deletion flow', async () => {


    // ⚠️ TODO: put a small test JPEG with EXIF metadata in a fixtures folder
    const photoPath = path.join(__dirname, 'fixtures', 'photo-with-exif.jpg');

    // Simulate EXIF-rich photo buffer in the EXIF service mock, if needed
    // (the actual file contents are less important than ensuring your strip
    // function is called and storage receives the sanitized buffer)

    const createRes = await request(app)
      .post('/api/reports')
      .set('Authorization', `Bearer ${userToken}`)
      .field('fireId', activeFire._id.toString())
      .field('description', 'Smoke and small flames observed near the ridge.')
      .attach('photo', photoPath); // form field name must match your route handler

    expect(createRes.status).toBe(201);
    expect(createRes.body).toHaveProperty('report');
    const reportId = createRes.body.report._id;

    // Expect report is initially queued for moderation
    expect(createRes.body.report.status).toBe('pending');
    expect(createRes.body.report.fire.toString()).toBe(activeFire._id.toString());

    // EXIF stripping must be performed before storage
    expect(exifService.stripExifMetadata).toHaveBeenCalledTimes(1);
    expect(storage.savePhoto).toHaveBeenCalledTimes(1);

    // We can also ensure that the stored report does NOT contain EXIF fields
    const storedReport = await Report.findById(reportId).lean();
    expect(storedReport).toBeTruthy();
    expect(storedReport.media).toBeDefined();
    expect(storedReport.media.url).toContain('sanitized-photo.jpg');

    // These fields should not exist or should be null if your schema had them:
    expect(storedReport.media.exif).toBeUndefined();
    expect(storedReport.media.gpsLatitude).toBeUndefined();
    expect(storedReport.media.gpsLongitude).toBeUndefined();

    const publicBeforeModeration = await request(app)
      .get(`/api/fires/${activeFire._id.toString()}/reports`)
      .query({ visibility: 'public' }); // optional query flag if you use it
    expect(publicBeforeModeration.status).toBe(200);
    expect(Array.isArray(publicBeforeModeration.body.reports)).toBe(true);

    // No pending reports should be included in public listing
    const publicIdsBefore = publicBeforeModeration.body.reports.map((r) => r._id);
    expect(publicIdsBefore).not.toContain(reportId);


    const moderateRes = await request(app)
      .patch(`/api/reports/${reportId}/moderate`)
      .set('Authorization', `Bearer ${adminToken}`)
      .send({ status: 'approved' });

    expect(moderateRes.status).toBe(200);
    expect(moderateRes.body.report.status).toBe('approved');

    const publicAfterModeration = await request(app)
      .get(`/api/fires/${activeFire._id.toString()}/reports`)
      .query({ visibility: 'public' });

    expect(publicAfterModeration.status).toBe(200);
    const publicReportsAfter = publicAfterModeration.body.reports;

    const approved = publicReportsAfter.find((r) => r._id === reportId);
    expect(approved).toBeDefined();
    expect(approved.status).toBe('approved');

    // Ensure no sensitive location/personal data is exposed publicly
    // (Assumes your public DTO omits EXIF/GPS fields entirely)
    expect(approved.media).toBeDefined();
    expect(approved.media).not.toHaveProperty('exif');
    expect(approved.media).not.toHaveProperty('gpsLatitude');
    expect(approved.media).not.toHaveProperty('gpsLongitude');
    expect(approved.media.url).toMatch(/^https?:\/\//);

    const deleteRes = await request(app)
      .delete(`/api/reports/${reportId}`)
      .set('Authorization', `Bearer ${userToken}`);

    expect(deleteRes.status).toBe(204);

    // Report should be removed from DB
    const afterDelete = await Report.findById(reportId);
    expect(afterDelete).toBeNull();

    // Media should be removed from storage and cache
    expect(storage.deletePhoto).toHaveBeenCalledTimes(1);
    expect(storage.deletePhoto).toHaveBeenCalledWith(
      storedReport.media.key // assuming media.key is what you store for deletion
    );

    // Deleted report should NOT appear in public listing
    const publicAfterDelete = await request(app)
      .get(`/api/fires/${activeFire._id.toString()}/reports`)
      .query({ visibility: 'public' });

    expect(publicAfterDelete.status).toBe(200);
    const publicIdsAfterDelete = publicAfterDelete.body.reports.map((r) => r._id);
    expect(publicIdsAfterDelete).not.toContain(reportId);
  });
});
