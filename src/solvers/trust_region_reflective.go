package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	trustRegionReflectiveMaxSteps  = 1000
	trustRegionReflectiveTolerance = 1e-6
	trustRegionReflectiveRadius    = 1.0
	trustRegionReflectiveMaxRadius = 10.0
	trustRegionReflectiveMinRadius = 1e-8
	trustRegionReflectiveLower     = -5.0
	trustRegionReflectiveUpper     = 5.0
)

// TrustRegionReflective minimizuje konfigurisanu benchmark funkciju koristeći trust-region metod sa REFLEKSIJOM
// na granicama [-5,5]: korak koji predloži izlazak iz opsega se odbija od zida umesto da se prosto kliuje —
// razlika od trust_region_dogleg.go koji uopšte nema pojam granica, samo dogleg putanju unutar radius-a
// problem.Point je početna tačka pretrage
func TrustRegionReflective(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := trustRegionReflectiveMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := trustRegionReflectiveTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	radius := trustRegionReflectiveRadius
	if v, ok := problem.Payload["radius"].(float64); ok {
		radius = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	x := make([]float64, dimensions)
	for d := range x {
		x[d] = clamp(problem.Point[d], trustRegionReflectiveLower, trustRegionReflectiveUpper)
	}

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		hessDiag := numHessDiag(fn.Evaluate, x, hessDiagEps)

		p := make([]float64, dimensions)
		for d := range p {
			p[d] = -grad[d] / (math.Abs(hessDiag[d]) + 1e-8)
		}
		if pn := norm(p); pn > radius {
			for d := range p {
				p[d] = p[d] * radius / pn
			}
		}

		// reflektuj p na granicu [-5,5] umesto da ga prosto kliujemo — "reflective" karakteristika ovog metoda
		for d := range p {
			raw := x[d] + p[d]
			if raw < trustRegionReflectiveLower {
				raw = trustRegionReflectiveLower + (trustRegionReflectiveLower - raw)
			}
			if raw > trustRegionReflectiveUpper {
				raw = trustRegionReflectiveUpper - (raw - trustRegionReflectiveUpper)
			}
			raw = clamp(raw, trustRegionReflectiveLower, trustRegionReflectiveUpper)
			p[d] = raw - x[d]
		}

		modelReduction := -dot(grad, p)
		for d := range p {
			modelReduction -= 0.5 * hessDiag[d] * p[d] * p[d]
		}

		newPoint := make([]float64, dimensions)
		for d := range newPoint {
			newPoint[d] = x[d] + p[d]
		}

		rho := (fn.Evaluate(x) - fn.Evaluate(newPoint)) / (modelReduction + 1e-10)

		if rho > 0.75 {
			radius = math.Min(radius*2, trustRegionReflectiveMaxRadius)
		}
		if rho < 0.25 {
			radius = math.Max(radius*0.5, trustRegionReflectiveMinRadius)
		}
		if rho > 0 {
			x = newPoint
		}

		value := fn.Evaluate(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}
		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), false)
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
		trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "trust_region_reflective",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
