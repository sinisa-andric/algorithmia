package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	goosePopulation = 20
	gooseMaxSteps   = 1000
	gooseTolerance  = 1e-6
	gooseRangeLow   = -5.0
	gooseRangeHigh  = 5.0
)

// Goose minimizuje konfigurisanu benchmark funkciju koristeći Wild Goose Migration Algorithm: guske lete u V
// formaciji, svaka prati gusku ispred sebe i predvodnika; predvodnik nema koga da prati pa umesto toga vrši malu
// lokalnu pretragu oko sebe da ne bi ostao zamrznut
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Goose(problem models.Problem) (result models.Result, err error) {

	if len(problem.Point) == 0 {
		err = fmt.Errorf("starting point is required")
		return result, err
	}

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return result, err
	}

	population := goosePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := gooseMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := gooseTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	sortByValue := func(points [][]float64, vals []float64) {
		idx := make([]int, len(points))
		for i := range idx {
			idx[i] = i
		}
		for i := 1; i < len(idx); i++ {
			for j := i; j > 0 && vals[idx[j]] < vals[idx[j-1]]; j-- {
				idx[j], idx[j-1] = idx[j-1], idx[j]
			}
		}
		sortedPoints := make([][]float64, len(points))
		sortedVals := make([]float64, len(points))
		for i, k := range idx {
			sortedPoints[i] = points[k]
			sortedVals[i] = vals[k]
		}
		copy(points, sortedPoints)
		copy(vals, sortedVals)
	}

	geese := make([][]float64, population)
	values := make([]float64, population)
	for i := range geese {
		goose := make([]float64, dimensions)
		for d := range goose {
			goose[d] = gooseRangeLow + rand.Float64()*(gooseRangeHigh-gooseRangeLow)
		}
		geese[i] = goose
		values[i] = fn.Evaluate(goose)
	}
	sortByValue(geese, values)

	best := append([]float64(nil), geese[0]...)
	bestValue := values[0]

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		leader := geese[0]

		for d := range geese[0] {
			geese[0][d] = geese[0][d] + (rand.Float64()*2-1)*1.0*(1-float64(steps)/float64(maxSteps))
			geese[0][d] = clamp(geese[0][d], gooseRangeLow, gooseRangeHigh)
		}
		values[0] = fn.Evaluate(geese[0])

		for i := 1; i < population; i++ {
			predecessor := geese[i-1]
			for d := range geese[i] {
				geese[i][d] = geese[i][d] + rand.Float64()*(predecessor[d]-geese[i][d]) + rand.Float64()*(leader[d]-geese[i][d])
				geese[i][d] = clamp(geese[i][d], gooseRangeLow, gooseRangeHigh)
			}
			values[i] = fn.Evaluate(geese[i])
		}

		sortByValue(geese, values)
		// Bez ove provere bestValue je bezuslovno prepisivan trenutnom najboljom jedinkom u jatu svakog
		// koraka — pošto se ceo roj (uključujući lidera) pomera bez elitizma, to je dozvoljavalo da
		// globalni best regresira (pogorša se) iz koraka u korak, pa noImprove logika nikad nije mogla
		// pouzdano da detektuje konvergenciju
		if values[0] < bestValue {
			bestValue = values[0]
			best = append([]float64(nil), geese[0]...)
		}
		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}

		if math.Abs(bestValue-prevBestValue) < tolerance {
			noImprove++
		} else {
			noImprove = 0
		}
		if noImprove >= maxNoImprove {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "goose",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
