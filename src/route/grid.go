package route

import (
	"fmt"
	"math"
	"net/http"
	"strconv"

	"algorithmia/src/functions"

	"github.com/gin-gonic/gin"
)

const (
	functionGridDefaultResolution = 60
	functionGridMaxResolution     = 150
	// isti raspon koji koristi grid_search.go, radi konzistentnosti
	functionGridLower = -5.0
	functionGridUpper = 5.0
)

// FunctionGridResponse je 2D mreža izračunatih vrednosti benchmark funkcije preko [-5,5]x[-5,5], namenjena
// direktnom renderovanju konture na frontend-u (bez approved/message/response omotača ostalih ruta)
type FunctionGridResponse struct {
	Function   string      `json:"function"`
	Resolution int         `json:"resolution"`
	Domain     [2]float64  `json:"domain"`
	MinValue   float64     `json:"min_value"`
	MaxValue   float64     `json:"max_value"`
	Grid       [][]float64 `json:"grid"`
}

// functionGridAxisValues generiše resolution ravnomerno raspoređenih tačaka preko [lower,upper], istom formulom
// koju koristi grid_search.go (axisValues[i] = lower + i*(upper-lower)/(resolution-1))
func functionGridAxisValues(resolution int, lower, upper float64) []float64 {

	axisValues := make([]float64, resolution)
	for i := 0; i < resolution; i++ {
		if resolution == 1 {
			axisValues[i] = (lower + upper) / 2
		} else {
			axisValues[i] = lower + float64(i)*(upper-lower)/float64(resolution-1)
		}
	}

	return axisValues
}

// FunctionGridHandler vraća 2D mrežu izračunatih vrednosti benchmark funkcije preko [-5,5]x[-5,5] za renderovanje
// konture. Funkcije definisane samo za fiksnu dimenzionalnost različitu od 2 (npr. hartmann_3d, shekel, colville,
// hartmann_6d, powell) se odbijaju sa 400, jer ValidateDimension(2) za njih ne prolazi
func FunctionGridHandler(ctx *gin.Context) {

	name := ctx.Param("name")

	fn, err := functions.Get(name)
	if err != nil {
		ctx.JSON(http.StatusNotFound, gin.H{
			"approved": false,
			"message":  "function not found: " + name,
		})
		return
	}

	if err := fn.ValidateDimension(2); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"approved": false,
			"message":  "/grid podržava samo 2D funkcije (N-dimenzionalne i 2D kategorije evaluirane u N=2): " + err.Error(),
		})
		return
	}

	resolution := functionGridDefaultResolution
	if raw := ctx.Query("resolution"); raw != "" {
		parsed, parseErr := strconv.Atoi(raw)
		if parseErr != nil || parsed < 1 {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"approved": false,
				"message":  "resolution mora biti pozitivan ceo broj",
			})
			return
		}
		resolution = parsed
	}

	if resolution > functionGridMaxResolution {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"approved": false,
			"message":  fmt.Sprintf("resolution %d prelazi maksimum od %d", resolution, functionGridMaxResolution),
		})
		return
	}

	axisValues := functionGridAxisValues(resolution, functionGridLower, functionGridUpper)

	grid := make([][]float64, resolution)
	minValue := math.Inf(1)
	maxValue := math.Inf(-1)
	for i, x := range axisValues {
		row := make([]float64, resolution)
		for j, y := range axisValues {
			value := fn.Evaluate([]float64{x, y})
			row[j] = value
			if value < minValue {
				minValue = value
			}
			if value > maxValue {
				maxValue = value
			}
		}
		grid[i] = row
	}

	ctx.JSON(http.StatusOK, FunctionGridResponse{
		Function:   fn.Name,
		Resolution: resolution,
		Domain:     [2]float64{functionGridLower, functionGridUpper},
		MinValue:   minValue,
		MaxValue:   maxValue,
		Grid:       grid,
	})
}
