package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	trustRegionDoglegRadius      = 1.0
	trustRegionDoglegMaxRadius   = 10.0
	trustRegionDoglegMaxSteps    = 1000
	trustRegionDoglegTolerance   = 1e-6
	trustRegionDoglegHessianStep = 1e-5
)

// TrustRegionDogleg minimizuje konfigurisanu benchmark funkciju koristeći trust-region dogleg metod: korak se bira
// između Cauchy (najstrmiji spust) tačke i Newton-ove tačke tako da ostane unutar oblasti poverenja,
// čiji se radijus prilagođava na osnovu uspešnosti koraka
// problem.Point je početna tačka pretrage
func TrustRegionDogleg(problem models.Problem) (result models.Result, err error) {

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

	radius := trustRegionDoglegRadius
	if v, ok := problem.Payload["radius"].(float64); ok {
		radius = v
	}

	maxRadius := trustRegionDoglegMaxRadius
	if v, ok := problem.Payload["max_radius"].(float64); ok {
		maxRadius = v
	}

	maxSteps := trustRegionDoglegMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := trustRegionDoglegTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	steps := 0

	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(point)
		if norm(gradient) < tolerance {
			break
		}

		hessianDiag := diagonalHessian(fn.Evaluate, point, trustRegionDoglegHessianStep)

		hg := make([]float64, len(point))
		for i := range hg {
			hg[i] = hessianDiag[i] * gradient[i]
		}
		gHg := dot(gradient, hg)

		cauchy := make([]float64, len(point))
		if gHg > 1e-10 {
			scale := dot(gradient, gradient) / gHg
			for i := range cauchy {
				cauchy[i] = -scale * gradient[i]
			}
		} else {
			for i := range cauchy {
				cauchy[i] = -gradient[i]
			}
		}

		newton := make([]float64, len(point))
		for i := range newton {
			newton[i] = -gradient[i] / math.Max(hessianDiag[i], 1e-10)
		}

		var step []float64
		switch {
		case norm(newton) <= radius:
			step = newton
		case norm(cauchy) >= radius:
			cn := norm(cauchy)
			step = make([]float64, len(cauchy))
			for i := range step {
				step[i] = radius * cauchy[i] / cn
			}
		default:
			diff := make([]float64, len(cauchy))
			for i := range diff {
				diff[i] = newton[i] - cauchy[i]
			}
			a := dot(diff, diff)
			b := 2 * dot(cauchy, diff)
			c := dot(cauchy, cauchy) - radius*radius
			tau := 0.0
			if a > 1e-12 {
				disc := math.Max(0, b*b-4*a*c)
				tau = (-b + math.Sqrt(disc)) / (2 * a)
				tau = math.Max(0, math.Min(1, tau))
			}
			step = make([]float64, len(cauchy))
			for i := range step {
				step[i] = cauchy[i] + tau*diff[i]
			}
		}

		newPoint := make([]float64, len(point))
		for i := range point {
			newPoint[i] = point[i] + step[i]
		}

		predicted := -dot(gradient, step)
		for i := range step {
			predicted -= 0.5 * hessianDiag[i] * step[i] * step[i]
		}

		actual := fn.Evaluate(point) - fn.Evaluate(newPoint)

		rho := 0.0
		if math.Abs(predicted) > 1e-12 {
			rho = actual / predicted
		}

		if rho > 0.75 {
			radius = math.Min(2*radius, maxRadius)
		}
		if rho < 0.25 {
			radius *= 0.25
		}
		if rho > 0.1 {
			point = newPoint
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "trust_region_dogleg",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
