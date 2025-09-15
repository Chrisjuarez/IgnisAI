// backend/routes/topography.js
// Fetches topopgraphic data and sends it to the db 
const express = require('express');
const axios = require('axios');
const { createCanvas, loadImage } = require('canvas');
const Topography = require('../models/Topography');
require('dotenv').config();

const router = express.Router();
const MAPBOX_TOKEN = process.env.MAPBOX_ACCESS_TOKEN;
const OVERPASS_URL = 'https://overpass-api.de/api/interpreter';
const Z = 14;  // Zoom level for Terrain-RGB tiles

function lonLatToTile(lon, lat, z) {
    const xt = ((lon + 180) / 360) * 2**z;
    const yt = (
        1 -
        Math.log(Math.tan(lat * Math.PI/180) + 1/Math.cos(lat * Math.PI/180)) / Math.PI
    ) / 2 * 2**z;
    return { x: Math.floor(xt), y: Math.floor(yt) };
}

function decodeElev(r,g,b){
    return ((r*256*256 + g*256 + b) * 0.1) - 10000;
}

function computeSlope(grid, scale=1){
    const dzdx = (
        (grid[0][2] + 2*grid[1][2] + grid[2][2]) -
        (grid[0][0] + 2*grid[1][0] + grid[2][0])
    ) / (8 * scale);
    const dzdy = (
        (grid[2][0] + 2*grid[2][1] + grid[2][2]) -
        (grid[0][0] + 2*grid[0][1] + grid[0][2])
    ) / (8 * scale);
    return Math.atan(Math.sqrt(dzdx*dzdx + dzdy*dzdy)) * (180/Math.PI);
}

router.get('/', async (req, res) => {
    const lat = parseFloat(req.query.lat) || 34.05;
    const lon = parseFloat(req.query.lon) || -118.24;

    try {
        // 1) Get Terrain-RGB tile and sample elevation + slope
        const { x,y } = lonLatToTile(lon, lat, Z);
        const url = `https://api.mapbox.com/v4/mapbox.terrain-rgb/${Z}/${x}/${y}@2x.pngraw?access_token=${MAPBOX_TOKEN}`;
        const img = await loadImage(url);
        const cvs = createCanvas(img.width, img.height);
        const ctx = cvs.getContext('2d');
        ctx.drawImage(img,0,0);

        const cx = img.width/2, cy = img.height/2;
        const grid = [];
        for (let dy=-1; dy<=1; dy++) {
            const row = [];
            for (let dx=-1; dx<=1; dx++) {
                const [r,g,b] = ctx.getImageData(cx+dx, cy+dy, 1,1).data;
                row.push(decodeElev(r,g,b));
            }
            grid.push(row);
        }
        const elevation = grid[1][1];
        const slope     = computeSlope(grid, 30);

    // 2) Fetch nearest land-use tag from Overpass
        const overpassQ = `
            [out:json][timeout:25];
            (
                way(around:50,${lat},${lon})["landuse"];
                relation(around:50,${lat},${lon})["landuse"];
            );
        out tags 1;
        `;
        const osm = await axios.post(OVERPASS_URL, overpassQ, {
            headers: { 'Content-Type':'text/plain' }
        });
        const landuse = osm.data.elements[0]?.tags?.landuse || 'Unknown';

        // 3) Upsert into Mongo as GeoJSON
        const filter = { 'location.coordinates': [ lon, lat ] };
        const update = {
            $set: {
                location: { type: 'Point', coordinates: [ lon, lat ] },
                elevation,
                slope,
                vegetationType: landuse
            }
        };
        const opts = { upsert: true, new: true };
        const doc = await Topography.findOneAndUpdate(filter, update, opts);

        res.json({ message: 'ok', data: doc });
    }
    catch(err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
