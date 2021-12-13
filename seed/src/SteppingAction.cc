/// \file SteppingAction.cc
/// \brief Implementation of the SteppingAction class

#include "SteppingAction.hh"
#include "EventAction.hh"
#include "DetectorConstruction.hh"
#include "Analysis.hh"
#include "G4PhysicalConstants.hh"
#include<cmath>

#include "G4Step.hh"
// #include "G4Event.hh"
#include "G4RunManager.hh"
// #include "G4LogicalVolume.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::SteppingAction(
    const DetectorConstruction* detectorConstruction,
    EventAction* eventAction)
  : G4UserSteppingAction(),
    fDetConstruction(detectorConstruction),
    fEventAction(eventAction)
{}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::~SteppingAction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//TODO: Add muon specific production storage

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  // Collect energy and track length step by step

  // get volume of the current step
  auto volume = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume();
  
  // energy deposit
  auto edep = step->GetTotalEnergyDeposit();

  // fetch position, change this to auto?
  G4StepPoint* PostStep = step->GetPostStepPoint();
  G4double PostStepX = PostStep->GetPosition().x();
  // We assume the production is symmetric
  // G4double PostStepY = PostStep->GetPosition().y();
  G4double PostStepZ = PostStep->GetPosition().z();
  G4double Velocity = PostStep->GetVelocity();

  // step length
  G4double stepLength = 0.;
  if ( step->GetTrack()->GetDefinition()->GetPDGCharge() != 0. ) {
    stepLength = step->GetStepLength();
  }

  // Only fill if inside target volume
  if ( volume == fDetConstruction->GetTargetPV() ) {
    fEventAction->AddTarget(edep,stepLength);
    // get analysis manager
    auto analysisManager = G4AnalysisManager::Instance();
    // Filling the histogram
    analysisManager->FillH2(0, PostStepX, PostStepZ);
    // Frank-Tamm Weights
    G4double beta = Velocity / c_light;
    // Need to change so that n is fetched
    G4double n = 1.33;
    G4double weightt = 1. - 1. / pow((n * beta), 2.);
    G4double weightb = 1. - 1. / pow((n), 2.);
    G4double weight = 0;
    if (weightt > 0) {
      weight = weightt / weightb;
    }
    analysisManager->FillH2(1, stepLength, PostStepZ, weight);
  }
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
