const request = require('supertest');
const app = require('../app');
const User = require('../models/User');

describe('Authentication Integration Tests', () => {
  let server;
  
  beforeAll(async () => {
    server = app.listen(5001);
  });

  afterAll(async () => {
    await server.close();
  });

  beforeEach(async () => {
    await User.deleteMany({});
  });

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
    });

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
    });

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
    });
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
    });

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
    });

    it('TC-005: should reject invalid credentials', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'john.doe@example.com',
          password: 'wrongpassword'
        })
        .expect(401);

      expect(response.body.message).toBe('Invalid email or password');
    });
  });

  describe('POST /api/auth/forgot-password', () => {
    beforeEach(async () => {
      await request(app)
        .post('/api/auth/register')
        .send({
          fullName: 'John Doe',
          email: 'john.doe@example.com',
          phone: '(555) 123-4567',
          password: 'password123'
        });
    });

    it('TC-006: should generate reset token for valid email', async () => {
      const response = await request(app)
        .post('/api/auth/forgot-password')
        .send({
          email: 'john.doe@example.com'
        })
        .expect(200);

      expect(response.body.message).toBe('Password reset link sent to your email');
      
      // Verify reset token was set
      const user = await User.findOne({ email: 'john.doe@example.com' });
      expect(user.passwordResetToken).toBeDefined();
      expect(user.passwordResetExpires).toBeDefined();
    });
  });

  describe('Rate Limiting', () => {
    it('TC-007: should enforce rate limiting on login endpoint', async () => {
      const loginData = {
        email: 'test@example.com',
        password: 'password123'
      };

      // Make 11 requests rapidly
      const requests = Array(11).fill(null).map(() => 
        request(app)
          .post('/api/auth/login')
          .send(loginData)
      );

      const responses = await Promise.all(requests);
      
      // First 10 should be 401 (invalid creds), 11th should be 429 (rate limited)
      const statusCodes = responses.map(res => res.status);
      expect(statusCodes.filter(code => code === 429).length).toBeGreaterThan(0);
    });
  });
});