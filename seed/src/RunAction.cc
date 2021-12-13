/// \file RunAction.cc
/// \brief Implementation of the RunAction class

#include "RunAction.hh"
#include "Analysis.hh"

#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::RunAction()
: G4UserRunAction()
{ 
  // set printing event number per each event
  G4RunManager::GetRunManager()->SetPrintProgress(1);  
  // Create analysis manager
  // The choice of analysis technology is done via selectin of a namespace
  // in B4Analysis.hh
  auto analysisManager = G4AnalysisManager::Instance();
  G4cout << "Using " << analysisManager->GetType() << G4endl;
  // Create directories 
  //analysisManager->SetHistoDirectoryName("histograms");
  //analysisManager->SetNtupleDirectoryName("ntuple");
  analysisManager->SetVerboseLevel(1);
  analysisManager->SetNtupleMerging(false);
    // Note: merging ntuples is available only with Root output

  // Book histograms, ntuple
  //

  // Creating 1D histograms
  analysisManager->CreateH1(
    "Etarget","Edep in target", 1000, 1. * GeV, 1 * TeV, "GeV");
  analysisManager->CreateH1(
    "Ltarget","trackL in target", 10000, 0. * km, 10 * km);

  // Creating 2D histograms
  // IDs start again from zero
  analysisManager->CreateH2(
    "Target_XZ","Target X vs Z",
    1000, -2.5 * m, 2.5 * m,
    1000, -20 * m, 20 * m);
  analysisManager->CreateH2(
    "Target_TrackZ","Target Track vs Z",
    10000, 1 * nm, 100 * m,
    1000, -20 * m, 20 * m,
    "m", "m", "log10"
    );


  // Creating ntuple
  //
  analysisManager->CreateNtuple("Showers", "Edep and TrackL");
  analysisManager->CreateNtupleDColumn("Etarget");
  analysisManager->CreateNtupleDColumn("Ltarget");
  analysisManager->FinishNtuple();

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::~RunAction()
{
  delete G4AnalysisManager::Instance();  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::BeginOfRunAction(const G4Run*)
{ 
  // inform the runManager to save random number seed
  G4RunManager::GetRunManager()->SetRandomNumberStore(false);

  // Get analysis manager
  auto analysisManager = G4AnalysisManager::Instance();

  // Open an output file
  //
  G4String fileName = "Shower";
  analysisManager->OpenFile(fileName);

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::EndOfRunAction(const G4Run* run)
{
  // print histogram statistics
  //
  auto analysisManager = G4AnalysisManager::Instance();
  if ( analysisManager->GetH1(1) ) {
    G4cout << G4endl << " ----> print histograms statistic ";
    if(isMaster) {
      G4cout << "for the entire run " << G4endl << G4endl; 
    }
    else {
      G4cout << "for the local thread " << G4endl << G4endl; 
    }
    
    G4cout << " ETarget : mean = " 
       << G4BestUnit(analysisManager->GetH1(0)->mean(), "Energy") 
       << " rms = " 
       << G4BestUnit(analysisManager->GetH1(0)->rms(),  "Energy") << G4endl;

    G4cout << " LTarget : mean = " 
      << G4BestUnit(analysisManager->GetH1(1)->mean(), "Length") 
      << " rms = " 
      << G4BestUnit(analysisManager->GetH1(1)->rms(),  "Length") << G4endl;
  }
  // save histograms & ntuple
  //
  analysisManager->Write();
  analysisManager->CloseFile();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

