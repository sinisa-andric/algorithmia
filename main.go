package main

import (
	"log"
	"time"

	"algorithmia/src/docs"
	"algorithmia/src/route"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func main() {

	gin.SetMode(gin.ReleaseMode)
	router := gin.Default()
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"},
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: false,
		MaxAge:           12 * time.Hour,
	}))
	router.POST("/solve", route.SolveProblemHandler)
	router.GET("/defaults", route.DefaultsHandler)
	router.GET("/functions", route.FunctionsHandler)
	router.GET("/solvers", route.SolversHandler)
	router.GET("/functions/:name", route.FunctionHandler)
	router.GET("/functions/:name/grid", route.FunctionGridHandler)
	router.GET("/solvers/:name", route.SolverHandler)

	log.Printf("[Algorithmia] Servis pokrenut na :9000 — %d solvera, %d benchmark funkcija", len(docs.SolverDocs), len(docs.FunctionDocs))
	router.Run(":9000")

}
