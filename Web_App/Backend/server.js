const cors = require('cors')
const express = require('express')
const homeRouter = require('./home')
const app = express()

app.use(cors({
    origin: ['http://localhost:5500', 'http://127.0.0.1:5500']
}))

app.use(express.json())
app.use('/home', homeRouter)

app.listen(3000, () => {
    console.log('Listening')
})