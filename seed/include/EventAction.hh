/// \file EventAction.hh
/// \brief Definition of the EventAction class

#ifndef EventAction_h
#define EventAction_h 1

#include "G4UserEventAction.hh"
#include "globals.hh"

class RunAction;

/// Event action class
///
/// It defines data members to hold the energy deposit and track lengths
/// of charged particles in the Target:
/// - fEnergyTarget, fTrackLTarget
/// which are collected step by step via the functions
/// - AddTarget()

class EventAction : public G4UserEventAction
{
  public:
    EventAction();
    virtual ~EventAction();

    virtual void  BeginOfEventAction(const G4Event* event);
    virtual void    EndOfEventAction(const G4Event* event);
    
    void AddTarget(G4double de, G4double dl);
    
  private:
    G4double  fEnergyTarget;
    G4double  fTrackLTarget; 
};

// inline functions

inline void EventAction::AddTarget(G4double de, G4double dl) {
  fEnergyTarget += de; 
  fTrackLTarget += dl;
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif

    
