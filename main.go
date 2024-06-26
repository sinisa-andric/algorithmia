package main

import (
	"algorithmia/src/route"
	"net/http"

	"github.com/gin-gonic/gin"
)

func main() {

	gin.SetMode(gin.ReleaseMode)
	router := gin.Default()

	router.POST("/solve", route.SolveProblem)
	http.ListenAndServe(":9000", router)

}
