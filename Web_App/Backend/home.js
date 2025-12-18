const express = require('express')
const { spawn } = require('child_process')
const router = express.Router()

const path = require('path');
const dotenv = require('dotenv')
dotenv.config({ path: path.resolve(__dirname, '../../.env') });


router.post('', (req, res) => {
    const genres = req.body.genres;
    const years = req.body.years;

    const python = spawn('python', ['-u', '../../Recommender/movie_rec.py', JSON.stringify(genres), JSON.stringify(years)]);

    let output = '';

    python.stdout.on('data', (data) => {
        output += data.toString();
    });

    python.stderr.on('data', (data) => {
        console.error('Python error:', data.toString());
    });

    python.on('close', (code) => {
        console.log('Python exited with code', code);
        try {
            const result = JSON.parse(output);
            console.log(result)
            res.json({ message: result })
        } catch (err) {
            console.error('Error parsing JSON', err)
            res.status(500).send('Error parsing recommendations')
        }
    });
})

router.get('/poster', async (req, res) => {
    const { movie, year, id } = req.query;
    const moviesURL = `https://api.themoviedb.org/3/movie/${id}`;
    const tvURL = `https://api.themoviedb.org/3/tv/${id}`;
    const backupURL = `https://api.themoviedb.org/3/search/tv`;
    const API_KEY = process.env.API_KEY;

    const paramsMovie = new URLSearchParams({
        api_key: API_KEY,
        query: movie,
        year: year
    });

    try {
        let response = await fetch(`${moviesURL}?${paramsMovie}`)
        if (!response.ok) {
            response = await fetch(`${tvURL}?${paramsMovie}`);
        }

        if (!response.ok) {
            const paramsSearch = new URLSearchParams({
                api_key: API_KEY,
                query: movie,
                year: year
            });

            response = await fetch(`${backupURL}?${paramsSearch}`);
        }

        if (!response.ok) {
            console.log("Triple Fail")
        }
        
        let data = await response.json();
        if (Array.isArray(data.results)) {
            data = data.results[0];
        }
        
        const posterPath = data.poster_path;
        const backdropPath = data.backdrop_path;
        res.json({ posterPath, backdropPath })
    } catch (err) {
        res.status(500).json({ error: 'Failed to fetch poster'});
    }
});

module.exports = router