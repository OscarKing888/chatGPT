// MonsterCharacter.h

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
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

    // Health variable
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Monster")
    float Health;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AI")
    float SightRange;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AI")
    float AttackRange;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AI")
	TArray<AActor*> PatrolPoints;
    

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
