// MonsterCharacter.h

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "AIController.h"
#include "MonsterCharacter.generated.h"


// Add a new enumeration for custom movement states
UENUM(BlueprintType)
enum class EMonsterMovementState : uint8
{
    Idle,
    Roaming,
    Patrolling,
    Attacking,
    Corpse
};


UCLASS()
class GAMEPLAYLIB_API AMonsterCharacter : public ACharacter
{
    GENERATED_BODY()

public:
    AMonsterCharacter();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI Nav")
    TWeakObjectPtr<ANavigationData> PreferredNavData;

    // Health variable
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Monster")
    float Health;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AI")
    float SightRange;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AI")
    float AttackRange;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AI")
	TArray<AActor*> PatrolPoints;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AI")
	TObjectPtr<AActor> MoveToTargetActor;
    

    // Override the TakeDamage method
    virtual float TakeDamage(float DamageAmount, const FDamageEvent& DamageEvent, AController* EventInstigator, AActor* DamageCauser) override;

    // Add a new function to notify the AIController of the death
    UFUNCTION(BlueprintCallable, Category = "Monster")
    void NotifyDeath();

    UFUNCTION(BlueprintCallable, Category = "Monster")
    void Attack(AActor* AttackTarget);

    UFUNCTION(BlueprintNativeEvent, Category = "Monster")
    void OnAttack(AActor* Actor);

    // Getter for MovementState
    UFUNCTION(BlueprintCallable, Category = "Monster")
    EMonsterMovementState GetMovementState() const { return MovementState; }

protected:
    // Called when the game starts or when spawned
    virtual void BeginPlay() override;

private:
    // Add a new variable for the movement state
    EMonsterMovementState MovementState;
};


UCLASS(BlueprintType, Blueprintable)
class ACustomNavAIController : public AAIController
{
    GENERATED_BODY()

public:
    
    virtual void FindPathForMoveRequest(const FAIMoveRequest& MoveRequest, FPathFindingQuery& Query, FNavPathSharedPtr& OutPath) const override;
};