const request = require('supertest');
const mongoose = require('mongoose');
const app = require('../app');
const User = require('../models/User');

describe('Authentication Integration Tests', () => {
  let server;
  
  beforeAll(async () => {
    // Connect to test database
    const mongoUri = process.env.MONGODB_TEST_URI || 'mongodb://localhost:27017/ignisai_test';
    if (mongoose.connection.readyState === 0) {
      await mongoose.connect(mongoUri);
    }
    
    server = app.listen(5001);
  }, 30000); // 30 second timeout

  afterAll(async () => {
    await server.close();
    await mongoose.connection.close();
  }, 30000);

  beforeEach(async () => {
    // Clean up test data
    await User.deleteMany({});
  }, 10000); // 10 second timeout

  describe('POST /api/auth/register', () => {
    it('TC-001: should register new user successfully', async () => {
      const userData = {
        fullName: 'John Doe',
        email: 'john.doe@example.com',
        phone: '(555) 123-4567',
        password: 'password123'
      };

      const response = await request(app)
        .post('/api/auth/register')
        .send(userData)
        .expect(201);

      expect(response.body.message).toBe('User registered successfully');
      expect(response.body.user.email).toBe(userData.email);
      expect(response.body.token).toBeDefined();
      
      // Verify user exists in database
      const user = await User.findOne({ email: userData.email });
      expect(user).toBeTruthy();
      expect(user.password).not.toBe(userData.password); // Should be hashed
    }, 15000);

    it('TC-002: should reject duplicate email registration', async () => {
      // First registration
      await request(app)
        .post('/api/auth/register')
        .send({
          fullName: 'John Doe',
          email: 'john.doe@example.com',
          phone: '(555) 123-4567',
          password: 'password123'
        });

      // Duplicate registration attempt
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          fullName: 'Jane Doe',
          email: 'john.doe@example.com',
          phone: '(555) 987-6543',
          password: 'password456'
        })
        .expect(400);

      expect(response.body.message).toBe('Email already registered');
    }, 15000);

    it('TC-003: should reject invalid input data', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          fullName: '',
          email: 'invalid-email',
          password: '123'
        })
        .expect(400);

      expect(response.body.message).toBe('Invalid input data');
    }, 15000);
  });

  describe('POST /api/auth/login', () => {
    beforeEach(async () => {
      // Create test user
      await request(app)
        .post('/api/auth/register')
        .send({
          fullName: 'John Doe',
          email: 'john.doe@example.com',
          phone: '(555) 123-4567',
          password: 'password123'
        });
    }, 10000);

    it('TC-004: should login with valid credentials', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'john.doe@example.com',
          password: 'password123'
        })
        .expect(200);

      expect(response.body.message).toBe('Login successful');
      expect(response.body.user.email).toBe('john.doe@example.com');
      expect(response.body.token).toBeDefined();
    }, 15000);

    it('TC-005: should reject invalid credentials', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'john.doe@example.com',
          password: 'wrongpassword'
        })
        .expect(401);

      expect(response.body.message).toBe('Invalid email or password');
    }, 15000);
  });
});